/* 
* @Author: BlahGeek
* @Date:   2015-01-14
* @Last Modified by:   Y.H.Zhou
* @Last Modified time: 2015-06-17
*/

#include <iostream>
#include "./kdtree.h"
#include <algorithm>
#

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
    result.s[0]=fmin(a.s[0],b.s[0]);
    result.s[1]=fmin(a.s[1],b.s[1]);
    result.s[2]=fmin(a.s[2],b.s[2]);
    result.s[3]=fmax(a.s[3],b.s[3]);
    result.s[4]=fmax(a.s[4],b.s[4]);
    result.s[5]=fmax(a.s[5],b.s[5]);
    return result;
}
void KDTree::Node::calcAABB(){

    cl_int    status;
    /**Step 1: Getting platforms and choose an available one(first).*/
    cl_platform_id platform;
    //getPlatform(platform);
    clGetPlatformIDs(1, &platform, NULL);
    /**Step 2:Query the platform and choose the first GPU device if has one.*/
    //cl_device_id *devices=getCl_device_id(platform);
    cl_device_id *devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, NULL);

    /**Step 3: Create context.*/
    cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

    /**Step 4: Creating command queue associate with the context.*/
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    printf("file\n");
    /**Step 5: Create program object */
    const char *filename = "aabb.cl";
    std::string sourceStr="aabb.cl";
    //status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    /**Step 6: Build program. */
    status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);

    /**Step 7: Initial input,output for the host and create memory objects for the kernel*/   //6400*4
    const size_t global_work_size= geometries.size();  ///
    const size_t local_work_size={64};    ///256 PE
    int groupNUM=global_work_size/local_work_size;
    printf("buffer");
    cl_float8* output = new cl_float8[(global_work_size/local_work_size)];

    cl_mem buffer_geometries = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, geometries.size()* sizeof(cl_int4),&geometries, NULL);
    cl_mem buffer_points=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,points.size()* sizeof(cl_float3),&points,NULL);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , groupNUM * sizeof(cl_float8), NULL, NULL);
    printf("point\n");
    /**Step 8: Create kernel object */
    cl_kernel kernel = clCreateKernel(program,"calaabb", NULL);
    printf("sdasdsdfasdfas\n");
    /**Step 9: Sets Kernel arguments.*/
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),buffer_geometries);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),buffer_points);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputBuffer);
    /**Step 10: Running the kernel.*/
    cl_event enentPoint;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &enentPoint);
    clWaitForEvents(1,&enentPoint); ///wait
    clReleaseEvent(enentPoint);
            
    /**Step 11: Read the cout put back to host memory.*/
    status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,groupNUM * sizeof(cl_float8), output, 0, NULL, NULL);
    cl_float8 result=output[0];
    for(int i=0;i<global_work_size/local_work_size;i++){
        result=updateaabb(result,output[i]);
    }
    /**Step 12: Clean the resources.*/
    status = clReleaseKernel(kernel);//*Release kernel.
    status = clReleaseProgram(program);    //Release the program object.
    //status = clReleaseMemObject(inputBuffer);//Release mem object.
    status = clReleaseMemObject(outputBuffer);
    status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
    status = clReleaseContext(context);//Release context.
    free(buffer_points);
    free(buffer_geometries);
    //free(input);
    free(output);
    free(devices);

    box_start.s[0]=result.s[0];
    box_start.s[1]=result.s[1];
    box_start.s[2]=result.s[2];
    box_end.s[0]=result.s[3];
    box_end.s[1]=result.s[4];
    box_end.s[2]=result.s[5];
}

 void KDTree::Node::setaabbSize() {
    for(int d = 0 ; d < 3 ; d += 1) {
        box_start.s[d] = box_start.s[d] - 1e-3f;
        box_end.s[d] = box_end.s[d] + 1e-3f;
    }
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

        cl_float no_split_cost = TRIANGLE_INTERSECT_COST * now->geo_indexes.size();

        cl_float best_pos[3]; cl_float best_val[3];
        for(int d = 0 ; d < 3 ; d += 1) {
            auto result = now->findBestSplit(d);
            best_pos[d] = result.first;
            best_val[d] = result.second;
        }
        auto best_val_it = std::min_element(best_val, best_val + 3);
        int best_dimension = best_val_it - best_val;

        if(*best_val_it > no_split_cost) {
            continue;
        }

        if(best_pos[best_dimension] <= now->box_start.s[best_dimension] ||
        best_pos[best_dimension] >= now->box_end.s[best_dimension]) {
            continue;
        }

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
        now->geo_indexes.clear();

        now->left->calcMinMaxVals();
        activeList.push(now->left.get());
        // this->left->setBoxSize();
        //this->left->split();

        now->right->calcMinMaxVals();
        activeList.push(now->right.get());
        // this->right->setBoxSize();
        //this->right->split();
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
    this->root->calcMinMaxVals();
    this->root->setBoxSize();
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
