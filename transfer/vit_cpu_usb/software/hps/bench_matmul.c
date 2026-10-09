/* Portable single-thread CPU starting baseline. No MMIO or DMA.
 * Build on the HPS with its native compiler to obtain board timings.
 * Usage: bench_matmul CASE_DIRECTORY [REPETITIONS [CALLS_PER_SAMPLE [WARMUPS]]]
 * Inputs/golden bytes come from scripts/generate_vectors.py.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/utsname.h>
#include <math.h>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unrecorded"
#endif
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif
static volatile int64_t sink;

static int8_t clip(int64_t v) {
    return (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
}
static int8_t rescale(int32_t sum, unsigned q) {
    int64_t v = (int64_t)sum * q;
    /* Define floor division explicitly; do not depend on signed >> in C. */
    int64_t shifted = v >= 0 ? v / 65536 : -((-v + 65535) / 65536);
    return clip(shifted);
}

static NOINLINE void full(const int8_t *w, const int8_t *x, int8_t *y,
                         int k, int n, int m, unsigned q) {
    for (int r=0; r<k; ++r) for (int c=0; c<m; ++c) {
        int32_t sum=0;
        for (int p=0; p<n; ++p) sum += (int32_t)w[r*n+p] * x[p*m+c];
        y[r*m+c] = rescale(sum,q);
    }
}

static NOINLINE void rtl(const int8_t *w, const int8_t *x, int8_t *y,
                        int k, int n, int m, unsigned q) {
    memset(y,0,(size_t)k*m);
    for (int start=0; start<n; start+=16)
        for (int r=0; r<k; ++r) for (int c=0; c<m; ++c) {
            int32_t sum=0;
            for (int p=start; p<start+16; ++p) sum += (int32_t)w[r*n+p]*x[p*m+c];
            y[r*m+c] = clip((int)y[r*m+c] + rescale(sum,q));
        }
}

static double now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec*1e9 + t.tv_nsec;
}
static int cmp_double(const void *a, const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
static FILE *open_case(const char *dir, const char *name, const char *mode) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/%s",dir,name);
    if (n<0 || (size_t)n>=sizeof(path)) { fprintf(stderr,"Path too long\n"); exit(1); }
    FILE *f=fopen(path,mode);
    if (!f) { perror(path); exit(1); }
    return f;
}
static void read_int8(FILE *f,int8_t *a,size_t len) {
    for (size_t i=0;i<len;++i) {
        int v;
        if (fscanf(f,"%d",&v)!=1 || v < -128 || v > 127) {
            fprintf(stderr,"Invalid INT8 input\n"); exit(1);
        }
        a[i]=(int8_t)v;
    }
}
static void check(const char *dir,const char *file,const int8_t *y,size_t len) {
    FILE *f=open_case(dir,file,"rb");
    size_t bad=0;
    for (size_t i=0;i<len;++i) {
        int expected=fgetc(f);
        if (expected==EOF) { fprintf(stderr,"Truncated golden file\n"); exit(1); }
        if (expected != (unsigned char)y[i]) ++bad;
    }
    if (fgetc(f)!=EOF) { fprintf(stderr,"Oversized golden file\n"); exit(1); }
    fclose(f);
    if (bad) { fprintf(stderr,"%s: %zu mismatches\n",file,bad); exit(1); }
}
typedef void (*kernel)(const int8_t *,const int8_t *,int8_t *,int,int,int,unsigned);

static int positive_arg(const char *s, int allow_zero) {
    char *end; errno=0; long v=strtol(s,&end,10);
    if (errno || end==s || *end || v<(allow_zero?0:1) || v>100000) {
        fprintf(stderr,"Invalid repetition/group/warmup count\n"); exit(1);
    }
    return (int)v;
}

int main(int argc,char **argv) {
    if (argc<2 || argc>5) { fprintf(stderr,"Usage: %s CASE_DIR [REPS [CALLS_PER_SAMPLE [WARMUPS]]]\n",argv[0]); return 1; }
    int reps=argc>2?positive_arg(argv[2],0):100;
    int inner=argc>3?positive_arg(argv[3],0):1;
    int warmups=argc>4?positive_arg(argv[4],1):10;
    FILE *samples=NULL;
    const char *sample_path=getenv("BENCH_SAMPLES_PATH");
    if (sample_path) {
        samples=fopen(sample_path,"w");
        if (!samples) { perror(sample_path); return 1; }
        fprintf(samples,"contract,sample,calls_per_sample,group_ms,per_call_ms\n");
    }
    FILE *f=open_case(argv[1],"input.txt","r");
    int k,n,m; unsigned q;
    if (fscanf(f,"%d%d%d%u",&k,&n,&m,&q)!=4 || k<1 || n<1 || m<1 ||
        k>4096 || n>4096 || m>4096 || n%16 || q>65535) {
        fprintf(stderr,"Invalid dimensions/scale; N must be a multiple of 16\n"); return 1;
    }
    int8_t *w=malloc((size_t)k*n),*x=malloc((size_t)n*m),*y=malloc((size_t)k*m);
    double *times=malloc((size_t)reps*sizeof(double));
    if (!w || !x || !y || !times) { fprintf(stderr,"Allocation failed\n"); return 1; }
    read_int8(f,w,(size_t)k*n); read_int8(f,x,(size_t)n*m); fclose(f);
    struct utsname host;
    if (uname(&host)) { perror("uname"); return 1; }
    fprintf(stderr,"host=%s arch=%s kernel=%s compiler=%s flags=%s; single thread; compute only; warmups=%d calls_per_sample=%d; percentiles describe group averages if calls_per_sample>1\n",
            host.sysname,host.machine,host.release,__VERSION__,BUILD_FLAGS,warmups,inner);
    printf("case,contract,k,n,m,q16,repetitions,median_ms,p95_ms,min_ms,checksum,verified,calls_per_sample,warmups,mean_ms,stddev_ms,cv,macs,gmac_per_s\n");
    kernel kernels[2]={full,rtl};
    const char *names[2]={"full_accumulation","rtl_compatibility"};
    const char *goldens[2]={"expected.full.bin","expected.rtl.bin"};
    for (int mode=0;mode<2;++mode) {
        kernels[mode](w,x,y,k,n,m,q); check(argv[1],goldens[mode],y,(size_t)k*m);
        for (int i=0;i<warmups;++i) { kernels[mode](w,x,y,k,n,m,q); sink+=y[0]; }
        for (int i=0;i<reps;++i) {
            double start=now_ns();
            for (int j=0;j<inner;++j) { kernels[mode](w,x,y,k,n,m,q); sink+=y[0]; }
            double elapsed=(now_ns()-start)/1e6;
            times[i]=elapsed/inner;
            if (samples) fprintf(samples,"%s,%d,%d,%.9f,%.9f\n",names[mode],i,inner,elapsed,times[i]);
            sink+=y[(size_t)i%((size_t)k*m)];
        }
        check(argv[1],goldens[mode],y,(size_t)k*m);
        int64_t checksum=0;
        for (size_t i=0;i<(size_t)k*m;++i) checksum+=y[i];
        double mean=0,variance=0;
        for (int i=0;i<reps;++i) mean+=times[i]/reps;
        for (int i=0;i<reps;++i) variance+=(times[i]-mean)*(times[i]-mean)/reps;
        double stddev=sqrt(variance);
        double macs=(double)k*n*m;
        qsort(times,(size_t)reps,sizeof(double),cmp_double);
        double median=reps%2?times[reps/2]:(times[reps/2-1]+times[reps/2])/2;
        int p95=(95*reps+99)/100-1;
        printf("\"%s\",%s,%d,%d,%d,%u,%d,%.9f,%.9f,%.9f,%" PRId64 ",true,%d,%d,%.9f,%.9f,%.6f,%.0f,%.6f\n",
               argv[1],names[mode],k,n,m,q,reps,median,times[p95],times[0],checksum,
               inner,warmups,mean,stddev,mean>0?stddev/mean:0,macs,mean>0?macs/(mean*1e6):0);
    }
    free(times); free(y); free(x); free(w);
    if (samples) fclose(samples);
    return 0;
}
