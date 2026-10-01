#include <cmath>
#include <cstdint>
extern "C" void graph_support(const int64_t* pts,int n,const uint8_t* img,int h,int w,int extra,double* out){
 for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
  const int64_t x1=pts[i*2],y1=pts[i*2+1],dx=pts[j*2]-x1,dy=pts[j*2+1]-y1;
  int dist=(int)std::sqrt((double)(dx*dx+dy*dy));if(extra&&dist<=0)continue;int samples=dist+extra,count=0,total=0;
  double step=samples>1?1.0/(samples-1):0.;
  for(int k=0;k<samples;++k){double t=(k==samples-1&&samples>1)?1.:k*step;int x=(int)std::nearbyint(x1+t*dx),y=(int)std::nearbyint(y1+t*dy);if(x>=0&&x<w&&y>=0&&y<h){++total;if(img[y*w+x]>64)++count;}}
  if(total>0){out[i*n+j]=(double)count/total;out[j*n+i]=(double)count/total;}
 }
}
