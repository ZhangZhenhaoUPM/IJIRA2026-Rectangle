#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
extern "C" int extract_cycles(const int64_t* pts,const uint8_t* adj,int n,double min_area,double max_area,double ratio,double compact,double angle,int cap,int64_t* output,int64_t* nfailed){
 const double pi=3.14159265358979323846; int count=0;*nfailed=0;
 std::vector<std::vector<int>> nb(n);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)if(adj[i*n+j])nb[i].push_back(j);
 for(int i=0;i<n;++i)for(int j:nb[i]){if(j<=i)continue;
  for(int k:nb[j]){if(k<=i)continue;
   for(int l:nb[k]){if(l<=j||!adj[i*n+l])continue;
    int ids[4]={i,j,k,l};int mnS=0,mxS=0,mnD=0,mxD=0;
    for(int a=1;a<4;++a){
     int64_t s=pts[2*ids[a]]+pts[2*ids[a]+1],di=pts[2*ids[a]+1]-pts[2*ids[a]];
     if(s<pts[2*ids[mnS]]+pts[2*ids[mnS]+1])mnS=a;
     if(s>pts[2*ids[mxS]]+pts[2*ids[mxS]+1])mxS=a;
     if(di<pts[2*ids[mnD]+1]-pts[2*ids[mnD]])mnD=a;
     if(di>pts[2*ids[mxD]+1]-pts[2*ids[mxD]])mxD=a;
    }
    int order[4]={mnS,mnD,mxS,mxD};int64_t x[4],y[4];
    for(int a=0;a<4;++a){x[a]=pts[2*ids[order[a]]];y[a]=pts[2*ids[order[a]]+1];}
    int64_t twice=0;for(int a=0;a<4;++a){int b=(a+3)%4;twice+=x[a]*y[b]-y[a]*x[b];}
    double area=.5*std::abs((double)twice);if(area<min_area||area>max_area)continue;
    double lens[4],per=0,lo=1e100,hi=0;
    for(int a=0;a<4;++a){int b=(a+1)%4;int64_t dx=x[a]-x[b],dy=y[a]-y[b];lens[a]=std::sqrt((double)(dx*dx+dy*dy));per+=lens[a];lo=std::min(lo,lens[a]);hi=std::max(hi,lens[a]);}
    if(!(lo/hi>=ratio))continue;
    if(!(4*pi*area/(per*per)>=compact))continue;
    bool angles_ok=true;
    for(int a=0;a<4;++a){int b=(a+1)%4,c=(a+2)%4;int64_t ax=x[a]-x[b],ay=y[a]-y[b],bx=x[c]-x[b],by=y[c]-y[b];double co=(double)(ax*bx+ay*by)/(lens[a]*lens[b]);co=std::max(-1.,std::min(1.,co));double ang=std::acos(co);if(!(pi/2-angle<=ang&&ang<=pi/2+angle)){angles_ok=false;break;}}
    if(!angles_ok){++*nfailed;continue;}
    for(int a=0;a<4;++a){x[a]=pts[2*ids[a]];y[a]=pts[2*ids[a]+1];}
    int orient=0;
    for(int a=0;a<4;++a){int b=(a+1)%4,c=(a+2)%4;int64_t cross=(x[b]-x[a])*(y[c]-y[b])-(y[b]-y[a])*(x[c]-x[b]);orient|=cross>0?1:cross<0?2:3;}
    if(orient==3)continue;
    for(int a=0;a<4;++a){output[count*8+2*a]=pts[2*ids[order[a]]];output[count*8+2*a+1]=pts[2*ids[order[a]]+1];}
    if(++count>=cap)return count;
   }
  }
 }
 return count;
}
