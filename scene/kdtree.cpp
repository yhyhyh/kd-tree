/* 
* @Author: BlahGeek
* @Date:   2015-01-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-01-20
*/

#include <iostream>
#include "./kdtree.h"
#include <algorithm>
#include <fstream>

using namespace hp;

std::pair<cl_float, cl_float> KDTree::Node::triangleMinMax(cl_int4 geo, int dimension) {
    cl_float val[3];
    for(int i = 0 ; i < 3 ; i += 1)
        val[i] = points[geo.s[i]].s[dimension];
    cl_float min = *std::min_element(val, val+3);
    cl_float max = *std::max_element(val, val+3);
    return std::make_pair(min, max);
}


cl_float8 updateaabb(cl_float8 a,cl_float8 b){
    cl_float8 result;
    result.s[0]=(a.s[0]>b.s[0])?b.s[0]:a.s[0];
    result.s[1]=(a.s[1]>b.s[1])?b.s[1]:a.s[1];
    result.s[2]=(a.s[2]>b.s[2])?b.s[2]:a.s[2];
    result.s[3]=(a.s[3]>b.s[3])?a.s[3]:b.s[3];
    result.s[4]=(a.s[4]>b.s[4])?a.s[4]:b.s[4];
    result.s[5]=(a.s[5]>b.s[5])?a.s[5]:b.s[5];

    return result;
}

int getPlatform(cl_platform_id &platform)
{
    platform = NULL;//the chosen platform
    
    cl_uint numPlatforms;//the NO. of platforms
    cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        std::cout<<"Error: Getting platforms!"<<std::endl;
        return -1;
    }
    
    /**For clarity, choose the first available platform. */
    if(numPlatforms > 0)
    {
        cl_platform_id* platforms =
        (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
    }
    else
        return -1;
}

/**Step 2:Query the platform and choose the first GPU device if has one.*/
cl_device_id *getCl_device_id(cl_platform_id &platform)
{
    cl_uint numDevices = 0;
    cl_device_id *devices=NULL;
    cl_int    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices > 0) //GPU available.
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }
    return devices;
}
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    
    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }
        
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout<<"Error: failed to open file:\n"<<filename<<std::endl;
    return -1;
}
void KDTree::Node::calcArray(){
   
}

void KDTree::Node::setaabbSize() {
    for(int d = 0 ; d < 3 ; d += 1) {
        box_start.s[d] = box_start.s[d] - 1e-3f;
        box_end.s[d] = box_end.s[d] + 1e-3f;
    }
}

void KDTree::Node::calcAABB(){
    
    cl_int    status;
    //Step 1: Getting platforms and choose an available one(first).
    cl_platform_id platform;
    //getPlatform(platform);
    getPlatform(platform);
    //Step 2:Query the platform and choose the first GPU device if has one.
    //cl_device_id *devices=getCl_device_id(platform);
    cl_device_id *devices=getCl_device_id(platform);
    //Step 3: Create context.
    cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);
    //Step 4: Creating command queue associate with the context.
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    size_t global_work_size[]= {geo_indexes.size()};  ///
    size_t local_work_size[]={2};    ///256 PE
    int groupNUM=global_work_size[0]/local_work_size[0];
    int num=geo_indexes.size();
    //Step 5: Create program object
    const char *filename = "aabb.cl";
    std::string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
    
    
    
    cl_int4 * geometries_Array=new cl_int4[geometries.size()];
    cl_float3 * points_Array=new cl_float3[points.size()];
    cl_int * geo_indexes_Array=new cl_int[geo_indexes.size()];
    for(int i=0;i<geo_indexes.size();i++)
        geo_indexes_Array[i]=geo_indexes[i];
    for(int i=0;i<geometries.size();i++)
        geometries_Array[i]=geometries[i];
    for(int i=0;i<points.size();i++)
        points_Array[i]=points[i];
    
    
    if(geo_indexes.size()%2)
    {
        geo_indexes_Array[geo_indexes.size()]=geo_indexes_Array[geo_indexes.size()-1];
        hp_log("%d",geo_indexes.size());
        global_work_size[0]++;
        groupNUM++;
        num++;
    }
    
    
    //Step 6: Build program.
    status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);
    if(status!=CL_SUCCESS){
        size_t len;
        char buffer[8 * 1024];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }
    //Step 7: Initial input,output for the host and create memory objects for the kernel   //6400*4

    
    
    
    cl_float8* output = new cl_float8[(global_work_size[0]/local_work_size[0])];
    cl_mem buffer_geometries = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, geometries.size()* sizeof(cl_int4),geometries_Array, NULL);
    
    cl_mem buffer_points=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,points.size()* sizeof(cl_float3),points_Array,NULL);
    
    cl_mem buffer_index=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,geo_indexes.size()* sizeof(cl_int),geo_indexes_Array,NULL);
    
    //cl_mem buffer_num=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cl_int),&num,NULL);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , groupNUM * sizeof(cl_float8), NULL, NULL);
    
    //Step 8: Create kernel object
    cl_kernel kernel = clCreateKernel(program,"calaabb", NULL);
    //Step 9: Sets Kernel arguments.
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),&buffer_geometries);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),&buffer_points);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),&buffer_index);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputBuffer);
    //status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_num);
    
    //Step 10: Running the kernel.
    cl_event enentPoint;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    
    //Step 11: Read the cout put back to host memory.
    status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,groupNUM * sizeof(cl_float8), output, 0, NULL, NULL);
    
    clFinish(commandQueue);
    cl_float8 result;
    result.s[0]=output[0].s[0];
    result.s[1]=output[0].s[1];
    result.s[2]=output[0].s[2];
    result.s[3]=output[0].s[3];
    result.s[4]=output[0].s[4];
    result.s[5]=output[0].s[5];
    
    for(int i=0;i < groupNUM;i++){
        result=updateaabb(result,output[i]);
    }
    this->box_start.s[0]=result.s[0];
    this->box_start.s[1]=result.s[1];
    this->box_start.s[2]=result.s[2];
    this->box_end.s[0]=result.s[3];
    this->box_end.s[1]=result.s[4];
    this->box_end.s[2]=result.s[5];
    
    free(output);
    status = clReleaseKernel(kernel);//Release kernel.
    status = clReleaseProgram(program);    //Release the program object.
    //status = clReleaseMemObject(inputBuffer);//Release mem object.
    status = clReleaseMemObject(outputBuffer);
    status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
    status = clReleaseContext(context);//Release context.
    free(buffer_points);
    free(buffer_geometries);
    free(buffer_index);
    //free(input);
    free(devices);
}




void KDTree::Node::calcMinMaxVals() {
    for(int d = 0 ; d < 3 ; d += 1) {
        for(auto & geo_index: geo_indexes) {
            auto geo = geometries[geo_index];
            auto result = this->triangleMinMax(geo, d);
            min_vals[d].push_back(result.first);
            max_vals[d].push_back(result.second);
        }
        std::sort(min_vals[d].begin(), min_vals[d].end());
        std::sort(max_vals[d].begin(), max_vals[d].end());
    }
}

void KDTree::Node::setBoxSize() {
    for(int d = 0 ; d < 3 ; d += 1) {
        box_start.s[d] = min_vals[d].front() - 1e-3f;
        box_end.s[d] = max_vals[d].back() + 1e-3f;
    }
}

#define TRAVERSAL_COST 1.f
#define TRIANGLE_INTERSECT_COST 3.f


std::pair<cl_float, cl_float> KDTree::Node::findBestSplit(int dimension) {
    std::vector<cl_float> all_vals(geo_indexes.size() * 2);
    memcpy(all_vals.data(), min_vals[dimension].data(), geo_indexes.size() * sizeof(cl_float));
    memcpy(all_vals.data() + geo_indexes.size(), 
           max_vals[dimension].data(), 
           geo_indexes.size() * sizeof(cl_float));
    std::sort(all_vals.begin(), all_vals.end());

    cl_float best_cost = CL_FLT_MAX;
    cl_float best_pos = 0;

    auto min_it = min_vals[dimension].begin();
    auto max_it = max_vals[dimension].begin();

    int leftT = 0; int rightT = geo_indexes.size();
    for(auto & val: all_vals) {
        if(val <= box_start.s[dimension] || val >= box_end.s[dimension])
            continue;
        while(min_it != min_vals[dimension].end() && 
              *min_it <= val) {
            // hp_log("left shift %f", *min_it);
            min_it ++;
            leftT ++;
        }
        while(max_it != max_vals[dimension].end() &&
              *max_it < val) {
            // hp_log("right shift %f", *max_it);
            max_it ++;
            rightT --;
        }
        cl_float area_length = box_end.s[dimension] - box_start.s[dimension];
        cl_float left_length = val - box_start.s[dimension];
        // hp_log("find: %d; %d %d; %f", dimension, leftT, rightT, val);
        cl_float cost = TRAVERSAL_COST + TRIANGLE_INTERSECT_COST * 
                        (leftT * left_length / area_length + rightT * (1.f - left_length / area_length));
        if(cost < best_cost) {
            best_cost = cost;
            best_pos = val;
        }
    }

    return std::make_pair(best_pos, best_cost);
}

#define LEAF_GEOMETRIES 5

void KDTree::split() {
    
    int best_dimension;
    cl_float best_pos[3]; cl_float best_val[3];

    std::queue<Node *> activeList;
    Node * now;
    activeList.push(this->root.get());
    fprintf(stderr, "%d\n", activeList.size());

    while (!activeList.empty()){
        
        now = activeList.front();
        activeList.pop();
        if(now->geo_indexes.size() < LEAF_GEOMETRIES) {
            continue;
        }
        printf("%d  ",now->geo_indexes.size());

        // Process Large Nodes. 
        if (now->geo_indexes.size()>25){

            printf("Large Node. \n");
            printf("%f %f %f\n", now->box_start.s[0], now->box_start.s[1], now->box_start.s[2]);
            printf("%f %f %f\n", now->box_end.s[0], now->box_end.s[1], now->box_end.s[2]);
            cl_float no_split_cost = TRIANGLE_INTERSECT_COST * now->geo_indexes.size(); 
            
            for(int d = 0 ; d < 3 ; d += 1) {
                if ((now->box_end.s[d]-now->box_start.s[d])>(now->box_end.s[(d+1)%3]-now->box_start.s[(d+1)%3]) &&
                    (now->box_end.s[d]-now->box_start.s[d])>(now->box_end.s[(d+2)%3]-now->box_start.s[(d+2)%3])){

                    best_dimension = d;
                    best_pos[best_dimension] = (now->box_end.s[d]+now->box_start.s[d])/2;
                }
            }  
            //fprintf(stderr, "%d\n", now->left->geo_indexes.size());
            now->left = std::make_unique<KDTree::Node>(points, geometries);
            now->right = std::make_unique<KDTree::Node>(points, geometries);
            now->left->parent = now->right->parent = now;
            now->left->box_start = now->right->box_start = now->box_start;
            now->left->box_end = now->right->box_end = now->box_end;        

            now->left->box_end.s[best_dimension] = best_pos[best_dimension];
            now->right->box_start.s[best_dimension] = best_pos[best_dimension];     

            for(auto & geo_index: now->geo_indexes) {
                auto geo = now->geometries[geo_index];
                std::vector<cl_float3> this_points = {points[geo.s[0]], points[geo.s[1]], points[geo.s[2]]};
                if(!std::all_of(this_points.begin(), this_points.end(), [&](const cl_float3 & p) ->bool {
                    return p.s[best_dimension] > best_pos[best_dimension];
                })) {
                    now->left->geo_indexes.push_back(geo_index);
                    hp_log("best: %lf",points[geometries[geo_index].s[0]].s[0]);
                }
                if(!std::all_of(this_points.begin(), this_points.end(), [&](const cl_float3 & p) ->bool {
                    return p.s[best_dimension] < best_pos[best_dimension];
                })) now->right->geo_indexes.push_back(geo_index);
                // if(this->left->contain(geo, best_dimension)) this->left->geo_indexes.push_back(geo_index);
                // if(this->right->contain(geo, best_dimension)) this->right->geo_indexes.push_back(geo_index);
            }

        }

        // Process Small Nodes. 
        else {

            printf("Small Node. \n");
            cl_float no_split_cost = TRIANGLE_INTERSECT_COST * now->geo_indexes.size(); 

            cl_float best_pos[3]; cl_float best_val[3];
            for(int d = 0 ; d < 3 ; d += 1) {
                auto result = now->findBestSplit(d);
                best_pos[d] = result.first;
                best_val[d] = result.second;
            }
            auto best_val_it = std::min_element(best_val, best_val + 3);
            best_dimension = best_val_it - best_val;    

            if(*best_val_it > no_split_cost) {
                continue;
            }   

            if(best_pos[best_dimension] <= now->box_start.s[best_dimension] ||
            best_pos[best_dimension] >= now->box_end.s[best_dimension]) {
                continue;
            }
            //fprintf(stderr, "%d\n", now->left->geo_indexes.size());
            now->left = std::make_unique<KDTree::Node>(points, geometries);
            now->right = std::make_unique<KDTree::Node>(points, geometries);
            now->left->parent = now->right->parent = now;
            now->left->box_start = now->right->box_start = now->box_start;
            now->left->box_end = now->right->box_end = now->box_end;        

            now->left->box_end.s[best_dimension] = best_pos[best_dimension];
            now->right->box_start.s[best_dimension] = best_pos[best_dimension];     

            for(auto & geo_index: now->geo_indexes) {
                auto geo = now->geometries[geo_index];
                std::vector<cl_float3> this_points = {points[geo.s[0]], points[geo.s[1]], points[geo.s[2]]};
                if(!std::all_of(this_points.begin(), this_points.end(), [&](const cl_float3 & p) ->bool {
                    return p.s[best_dimension] > best_pos[best_dimension];
                })) now->left->geo_indexes.push_back(geo_index);
                if(!std::all_of(this_points.begin(), this_points.end(), [&](const cl_float3 & p) ->bool {
                    return p.s[best_dimension] < best_pos[best_dimension];
                })) now->right->geo_indexes.push_back(geo_index);
                // if(this->left->contain(geo, best_dimension)) this->left->geo_indexes.push_back(geo_index);
                // if(this->right->contain(geo, best_dimension)) this->right->geo_indexes.push_back(geo_index);
            }
        }

        now->geo_indexes.clear();
        //now->left->calcMinMaxVals();
        now->left->calcAABB();
        activeList.push(now->left.get());
        // this->left->setBoxSize();
        //now->left->setaabbSize();

        //now->right->calcMinMaxVals();
        now->right->calcAABB();
        activeList.push(now->right.get());
        // this->right->setBoxSize();
        //now->right->setaabbSize();
    }
}

void KDTree::Node::removeEmptyNode() {
    if(left == nullptr && right == nullptr) return;
    left->removeEmptyNode();
    right->removeEmptyNode();

    if(left && left->geo_indexes.size() == 0 && left->left == nullptr && left->right == nullptr) {
        this->box_start = right->box_start;
        this->box_end = right->box_end;
        this->geo_indexes = right->geo_indexes;
        this->left = std::move(right->left);
        hp_assert(right->left == nullptr);
        std::unique_ptr<KDTree::Node> tmp = std::move(right->right);
        hp_assert(right->right == nullptr);
        this->right = std::move(tmp);
        if(this->right)
            this->right->parent = this;
        if(this->left)
            this->left->parent = this;
        return;
    }
    if(right && right->geo_indexes.size() == 0 && right->left == nullptr && right->right == nullptr) {
        this->box_start = left->box_start;
        this->box_end = left->box_end;
        this->geo_indexes = left->geo_indexes;
        this->right = std::move(left->right);
        hp_assert(left->right == nullptr);
        std::unique_ptr<KDTree::Node> tmp = std::move(left->left);
        hp_assert(left->left == nullptr);
        this->left = std::move(tmp);
        if(this->right)
            this->right->parent = this;
        if(this->left)
            this->left->parent = this;
        return;
    }
}

int KDTree::Node::debugPrint(int depth, int id) {
    for(int i = 0 ; i < depth ; i += 1)
        fprintf(stderr, "  ");
    fprintf(stderr, "ID %d", id++);
    fprintf(stderr, "(%f %f %f)->(%f %f %f) ",
           box_start.s[0], box_start.s[1], box_start.s[2],
           box_end.s[0], box_end.s[1], box_end.s[2]);
    if(geo_indexes.size() != 0) {
        fprintf(stderr, " LEAF, size = %lu\n", geo_indexes.size());
        for(auto & geo_id: geo_indexes) {
            fprintf(stderr, "...");
            for(int i = 0 ; i < depth ; i += 1)
                fprintf(stderr, "  ");
            fprintf(stderr, "Triangle %d %d %d\n", geometries[geo_id].s[0],
                    geometries[geo_id].s[1], geometries[geo_id].s[2]);
        }
    }
    else {
        fprintf(stderr, "\n");
        if(left)
            id = left->debugPrint(depth + 1, id);
        if(right)
            id = right->debugPrint(depth + 1, id);
    }
    return id;
}

KDTree::KDTree(std::string filename): Scene(filename) {
    this->root = std::make_unique<KDTree::Node>(this->points, this->geometries);
    for(size_t i = 0 ; i < this->geometries.size() ; i += 1)
        this->root->geo_indexes.push_back(i);
    this->root->calcAABB();
    this->root->setaabbSize();
    //this->root->calcMinMaxVals();
    //this->root->setBoxSize();
    this->split();
    this->root->removeEmptyNode();

    this->root->debugPrint();
}

std::pair<std::vector<KDTreeNodeHeader>, std::vector<cl_int>>
    KDTree::getData() {

        std::vector<KDTree::Node *> nodes;
        std::map<KDTree::Node *, int> nodes_map;
        std::function<void(KDTree::Node * node)> walk;
        walk = [&](KDTree::Node * node) {
            nodes_map[node] = nodes.size();
            nodes.push_back(node);
            if(node->left) walk(node->left.get());
            if(node->right) walk(node->right.get());
        };
        walk(this->root.get());

        std::vector<KDTreeNodeHeader> header_data;
        std::vector<cl_int> triangle_data;

        for(size_t i = 0 ; i < nodes.size() ; i += 1) {
            auto node = nodes[i];
            KDTreeNodeHeader header;
            header.data = -1;
            if(node->geo_indexes.size() > 0) {
                header.data = triangle_data.size();
                triangle_data.push_back(node->geo_indexes.size());
                for(auto & x: node->geo_indexes)
                    triangle_data.push_back(x);
            }
            header.box_start = node->box_start;
            header.box_end = node->box_end;
            header.child = (node->left == nullptr) ? -1 : 
                            nodes_map[node->left.get()];
            header.parent = (node->parent == nullptr) ? -1 :
                            nodes_map[node->parent];
            header.sibling = -1;
            if(node->parent != nullptr && node->parent->right != nullptr &&
               node->parent->right.get() != node) {
                header.sibling = nodes_map[node->parent->right.get()];
            }

            header_data.push_back(header);
        }

        return std::make_pair(header_data, triangle_data);

    }
