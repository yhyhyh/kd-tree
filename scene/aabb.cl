inline float _calmax(float p1, float p2, float p3){
    float result;
    if(p1>p2)
        result=p1;
    else 
        result=p2;
    if(result>p3)
        result=result;
    else
        result=p3;
    return result;
}
inline float _calmin(float p1, float p2, float p3){
    float result;
    if(p1>p2)
        result=p2;
    else 
        result=p1;
    if(result>p3)
        result=p3;
    else
        result=result;
    return result;
}
inline float8 _handleone(int4 geometries,__global float3 * points){
    int point1=geometries.s0;
    int point2=geometries.s1;
    int point3=geometries.s2;
    
    float ax=_calmin(points[point1][0],points[point2][0],points[point3][0]);
    float ay=_calmin(points[point1][1],points[point2][1],points[point3][1]);
    float az=_calmin(points[point1][2],points[point2][2],points[point3][2]);
    float bx=_calmax(points[point1][0],points[point2][0],points[point3][0]);
    float by=_calmax(points[point1][1],points[point2][1],points[point3][1]);
    float bz=_calmax(points[point1][2],points[point2][2],points[point3][2]);
    float8 result;
    result.s0=ax;
    result.s1=ay;
    result.s2=az;
    result.s3=bx;
    result.s4=by;
    result.s5=bz;
    
   // printf("result:%lf\n",result.s0);
    return result;
    
}
inline float8 _updateaabb(float8 a,float8 b){
    float8 result;
    result.s0=fmin(a.s0,b.s0);
    result.s1=fmin(a.s1,b.s1);
    result.s2=fmin(a.s2,b.s2);
    result.s3=fmax(a.s3,b.s3);
    result.s4=fmax(a.s4,b.s4);
    result.s5=fmax(a.s5,b.s5);
    //printf("a %f\tb %f\n",a.s0,b.s0);
    return result;
}
__kernel void calaabb(__global int4 *geometries,__global float3 * points,__global int * geo_indexes,__global float8 *output){
    __local float8 resArray[2];
    unsigned int tid = get_local_id(0); 
    unsigned int bid = get_group_id(0); 
    unsigned int i = get_global_id(0);
    float8 res;
    unsigned int localSize = get_local_size(0); 
    unsigned int globalSize= get_global_size(0);
    printf("num:%d\n",sizeof(geo_indexes));
    
    res.s0=points[geometries[geo_indexes[0]].s0].s0;
    res.s1=points[geometries[geo_indexes[0]].s0].s1;
    res.s2=points[geometries[geo_indexes[0]].s0].s2;
    res.s3=points[geometries[geo_indexes[0]].s0].s0;
    res.s4=points[geometries[geo_indexes[0]].s0].s1;
    res.s5=points[geometries[geo_indexes[0]].s0].s2;
    
    while(i<num){
        float8 tmp=_handleone(geometries[geo_indexes[i]],points);
        res=_updateaabb(tmp,res);
        i+=globalSize;
}
    resArray[tid]=res;
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s = localSize >> 1;s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
           // resArray[tid] += resArray[tid + s];
            resArray[tid]=_updateaabb(resArray[tid],resArray[tid+s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0){
        output[bid] = resArray[0];
        printf("%d\t%lf\n",bid,output[2].s0);
        //output[bid].s0=0;
    }
}
