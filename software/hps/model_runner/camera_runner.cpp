// Persistent CPU-only camera service. Protocol v1, little-endian binary stdio.
// Request: "VIT1", uint32 id, 3*224*224 float32 RGB CHW values.
// Ready: "RDY1", float64 load_ms.
// Reply: "RES1", uint32 id, float64 pack_ms, infer_ms, service_ms, 1000 float32 logits.
// service_ms starts after the request payload arrives; excludes transfer/write time.
#include <net.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
static void read_exact(char* p, size_t n) {
    if (!std::cin.read(p,n)) throw std::runtime_error("Truncated request");
}
template<class T> static void send(const T& value) {
    std::cout.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: camera_runner MODEL.param MODEL.bin THREADS INPUT_BLOB OUTPUT_BLOB fp32|runtime-int8\n";
        return 2;
    }
    try {
        uint16_t endian=1;
        if (*reinterpret_cast<char*>(&endian)!=1 || sizeof(float)!=4 || sizeof(double)!=8)
            throw std::runtime_error("Protocol requires little-endian float32/float64 host");
        const std::string threads_arg=argv[3], precision=argv[6];
        if (threads_arg!="1" && threads_arg!="2") throw std::runtime_error("THREADS must be 1 or 2");
        if (precision!="fp32" && precision!="runtime-int8") throw std::runtime_error("Invalid precision");
        ncnn::Net net;
        net.opt.num_threads=std::stoi(threads_arg);
        net.opt.use_vulkan_compute=false;
        net.opt.use_fp16_packed=false;
        net.opt.use_fp16_storage=false;
        net.opt.use_fp16_arithmetic=false;
        net.opt.use_bf16_storage=false;
        net.opt.use_int8_inference=(precision=="runtime-int8");
        auto start=Clock::now();
        if (net.load_param(argv[1]) || net.load_model(argv[2])) throw std::runtime_error("Model load failed");
        double load_ms=elapsed(start);
        std::cout.write("RDY1",4); send(load_ms); std::cout.flush();
        if (!std::cout) throw std::runtime_error("Ready write failed");
        std::vector<float> raw(3*224*224);
        while (true) {
            char magic[4];
            std::cin.read(magic,4);
            if (std::cin.gcount()==0 && std::cin.eof()) return 0;
            if (std::cin.gcount()!=4 || std::memcmp(magic,"VIT1",4)) throw std::runtime_error("Invalid request header");
            uint32_t id; read_exact(reinterpret_cast<char*>(&id),sizeof(id));
            read_exact(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(float));
            auto service_start=Clock::now();
            ncnn::Mat x(224,224,3);
            if (x.empty()) throw std::runtime_error("Input allocation failed");
            for (int c=0;c<3;c++) {
                float* dst=x.channel(c);
                for (int i=0;i<224*224;i++) {
                    float v=raw[c*224*224+i];
                    if (!std::isfinite(v)) throw std::runtime_error("Nonfinite input");
                    dst[i]=v;
                }
            }
            double pack_ms=elapsed(service_start);
            start=Clock::now();
            ncnn::Extractor ex=net.create_extractor();
            if (ex.input(argv[4],x)) throw std::runtime_error("Input blob binding failed");
            ncnn::Mat y;
            if (ex.extract(argv[5],y)) throw std::runtime_error("Output extraction failed");
            double infer_ms=elapsed(start);
            if (y.dims!=1 || y.w!=1000 || y.elempack!=1 || y.elemsize!=sizeof(float))
                throw std::runtime_error("Expected unpacked FP32 [1000] logits");
            for (int i=0;i<1000;i++) if (!std::isfinite(y[i])) throw std::runtime_error("Nonfinite logits");
            double service_ms=elapsed(service_start);
            std::cout.write("RES1",4); send(id); send(pack_ms); send(infer_ms); send(service_ms);
            std::cout.write(reinterpret_cast<const char*>(static_cast<const float*>(y)),4000);
            std::cout.flush();
            if (!std::cout) throw std::runtime_error("Response write failed");
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
