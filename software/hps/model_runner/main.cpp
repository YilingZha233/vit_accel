// Native CPU inference adapter. Requires a converted, validated ncnn model.
// Input: little-endian FP32 RGB CHW [3,224,224], preprocessed on desktop.
#include <net.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
static std::string quote(const std::string& s) {
    std::string out="\"";
    for (unsigned char c:s) {
        if(c=='"' || c=='\\') { out+='\\'; out+=c; }
        else if(c<32) throw std::runtime_error("Control character in path/id");
        else out+=c;
    }
    return out+'"';
}
static int number(const char* value, int low) {
    size_t n; int v=std::stoi(value,&n);
    if(n!=std::string(value).size() || v<low || v>1000000) throw std::runtime_error("Invalid numeric argument");
    return v;
}
struct Entry { std::string id, path; int label; };
static ncnn::Mat input(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    std::vector<float> raw(3*224*224);
    if(!f.read(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(float)) || f.peek()!=EOF)
        throw std::runtime_error("Invalid input bytes: "+path);
    ncnn::Mat x(224,224,3);
    for(int c=0;c<3;c++) {
        float* dst=x.channel(c);
        for(int i=0;i<224*224;i++) {
            float v=raw[c*224*224+i];
            if(!std::isfinite(v)) throw std::runtime_error("Nonfinite input");
            dst[i]=v;
        }
    }
    return x;
}
static ncnn::Mat infer(ncnn::Net& net,const ncnn::Mat& x,const char* in,const char* out) {
    ncnn::Extractor ex=net.create_extractor();
    if(ex.input(in,x)) throw std::runtime_error("Input blob binding failed");
    ncnn::Mat y;
    if(ex.extract(out,y)) throw std::runtime_error("Output extraction failed");
    return y;
}
static void validate(const ncnn::Mat& y) {
    if(y.dims!=1 || y.w!=1000 || y.elempack!=1 || y.elemsize!=sizeof(float))
        throw std::runtime_error("Expected unpacked FP32 [1000] logits; inspect converted output layout");
    for(int i=0;i<1000;i++) if(!std::isfinite(y[i])) throw std::runtime_error("Nonfinite logits");
}
static long rss_kib() {
    std::ifstream f("/proc/self/status"); std::string line;
    while(std::getline(f,line)) if(line.find("VmRSS:")==0) {
        std::istringstream s(line.substr(6)); long value; if(s>>value) return value;
    }
    return -1;
}
int main(int argc,char** argv) {
    if(argc!=13) {
        std::cerr<<"Usage: model_runner eval|bench MODEL.param MODEL.bin INPUTS.tsv NEW_OUT THREADS WARMUPS RUNS INPUT_BLOB OUTPUT_BLOB MIN_SECONDS fp32|runtime-int8\n";
        return 2;
    }
    try {
        unsigned short endian=1;
        if(*reinterpret_cast<unsigned char*>(&endian)!=1) throw std::runtime_error("Little-endian host required");
        std::string mode=argv[1], outdir=argv[5], precision=argv[12];
        if(mode!="eval" && mode!="bench") throw std::runtime_error("Invalid mode");
        if(precision!="fp32" && precision!="runtime-int8") throw std::runtime_error("Invalid precision");
        int threads=number(argv[6],1), warmups=number(argv[7],0), runs=number(argv[8],1);
        int seconds=number(argv[11],0);
        std::string tsv=argv[4]; auto slash=tsv.find_last_of('/');
        std::string base=slash==std::string::npos?".":tsv.substr(0,slash);
        std::ifstream file(tsv.c_str()); if(!file) throw std::runtime_error("Cannot open TSV");
        std::vector<Entry> entries; std::string line;
        while(std::getline(file,line)) {
            std::istringstream s(line); Entry e; std::string label;
            if(!std::getline(s,e.id,'\t') || !std::getline(s,label,'\t') || !std::getline(s,e.path))
                throw std::runtime_error("TSV requires id TAB label TAB input path; no header");
            if(e.id.empty() || e.path.empty()) throw std::runtime_error("Empty id/path");
            quote(e.id); e.label=number(label.c_str(),0);
            if(e.label>=1000) throw std::runtime_error("Invalid label");
            if(e.path[0]!='/') e.path=base+"/"+e.path;
            entries.push_back(e);
        }
        if(entries.empty() || (mode=="bench" && entries.size()>32)) throw std::runtime_error("Need inputs; bench preloads at most 32");
        if(mkdir(outdir.c_str(),0775)!=0) throw std::runtime_error("Output must be a new directory with an existing parent");
        ncnn::Net net;
        net.opt.num_threads=threads;
        net.opt.use_vulkan_compute=false;
        net.opt.use_fp16_packed=false;
        net.opt.use_fp16_storage=false;
        net.opt.use_fp16_arithmetic=false;
        net.opt.use_bf16_storage=false;
        net.opt.use_int8_inference=(precision=="runtime-int8");
        auto start=Clock::now();
        if(net.load_param(argv[2]) || net.load_model(argv[3])) throw std::runtime_error("Model load failed");
        double load_ms=ms(start); long loaded_rss=rss_kib();
        ncnn::Mat first_input=input(entries[0].path);
        start=Clock::now(); auto first=infer(net,first_input,argv[9],argv[10]); double first_ms=ms(start); validate(first);
        for(int i=0;i<warmups;i++) validate(infer(net,first_input,argv[9],argv[10]));
        std::vector<ncnn::Mat> cached;
        if(mode=="bench") for(const auto& e:entries) cached.push_back(input(e.path));
        std::ofstream samples((outdir+"/latency.csv").c_str());
        samples<<"sample,input_index,model_ms\n"<<std::setprecision(12);
        std::vector<double> times; int completed=0, correct1=0, correct5=0;
        auto loop_start=Clock::now();
        while(mode=="eval" ? completed<int(entries.size()) : (completed<runs || ms(loop_start)<seconds*1000.)) {
            int i=completed%entries.size();
            ncnn::Mat x=mode=="eval"?input(entries[i].path):cached[i];
            start=Clock::now(); auto y=infer(net,x,argv[9],argv[10]); double elapsed=ms(start);
            validate(y); times.push_back(elapsed); samples<<completed<<","<<i<<","<<elapsed<<"\n";
            if(mode=="eval") {
                char name[64]; std::snprintf(name,sizeof(name),"/%05d.logits.bin",i);
                std::ofstream f((outdir+name).c_str(),std::ios::binary);
                f.write(reinterpret_cast<const char*>(static_cast<const float*>(y)),1000*sizeof(float));
                if(!f) throw std::runtime_error("Output write failed");
                std::vector<int> idx(1000); for(int j=0;j<1000;j++) idx[j]=j;
                std::partial_sort(idx.begin(),idx.begin()+5,idx.end(),[&](int a,int b){return y[a]>y[b];});
                correct1+=idx[0]==entries[i].label;
                correct5+=std::find(idx.begin(),idx.begin()+5,entries[i].label)!=idx.begin()+5;
            }
            completed++;
        }
        double service_s=ms(loop_start)/1000.;
        if(!samples) throw std::runtime_error("Sample write failed");
        std::ofstream ids((outdir+"/image_ids.json").c_str()); ids<<"[";
        for(size_t i=0;i<entries.size();i++) ids<<(i?",":"")<<quote(entries[i].id);
        ids<<"]\n";
        double sum=0; for(double t:times) sum+=t;
        double mean=sum/times.size(), var=0; for(double t:times) var+=(t-mean)*(t-mean)/times.size();
        std::sort(times.begin(),times.end()); size_t n=times.size();
        struct rusage usage; if(getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
        double peak_mib=usage.ru_maxrss/1048576.;
#else
        double peak_mib=usage.ru_maxrss/1024.;
#endif
        std::ofstream report((outdir+"/run.json").c_str());
        struct utsname host; if(uname(&host)) throw std::runtime_error("uname failed");
        report<<std::setprecision(12)<<"{\n\"mode\":"<<quote(mode)<<",\"precision_request\":"<<quote(precision)
              <<",\"platform_machine\":"<<quote(host.machine)<<",\"platform_system\":"<<quote(host.sysname)
              <<",\"int8_kernel_coverage\":\"not verified by this runner\",\"threads\":"<<threads
              <<",\"completed\":"<<completed<<",\"failed\":0,\"warmups\":"<<warmups
              <<",\"load_ms\":"<<load_ms<<",\"first_inference_ms\":"<<first_ms
              <<",\"mean_ms\":"<<mean<<",\"median_ms\":"<<(n%2?times[n/2]:(times[n/2-1]+times[n/2])/2)
              <<",\"p95_ms\":"<<times[size_t(std::ceil(.95*n))-1]<<",\"min_ms\":"<<times.front()
              <<",\"max_ms\":"<<times.back()<<",\"stddev_ms\":"<<std::sqrt(var)<<",\"cv\":"<<std::sqrt(var)/mean
              <<",\"images_per_inference_second\":"<<1000/mean<<",\"service_loop_seconds\":"<<service_s
              <<",\"service_loop_images_per_second\":"<<completed/service_s
              <<",\"peak_process_rss_mib\":"<<peak_mib<<",\"loaded_rss_kib\":"<<loaded_rss
              <<",\"end_rss_kib\":"<<rss_kib()<<",\"top1_correct\":"<<correct1<<",\"top5_correct\":"<<correct5
              <<",\"accuracy_available\":"<<(mode=="eval"?"true":"false")
              <<",\"scope\":\"extractor creation, input binding, inference and output extraction; preprocessed inputs; CPU only\"}\n";
        if(!report || !ids) throw std::runtime_error("Metadata write failed");
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
