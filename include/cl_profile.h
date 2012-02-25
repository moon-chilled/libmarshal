#ifndef LIBMARSHAL_CL_PROFILE_H_
#define LIBMARSHAL_CL_PROFILE_H_

#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#include "local_cl.hpp"
namespace {
class Profiling {
 public:
  Profiling(cl::CommandQueue q, std::string id): id_(id),
    profiling_status_(CL_SUCCESS), queue_(q) {
    clRetainCommandQueue(q());
    profiling_status_ = clSetCommandQueueProperty(queue_(),
        CL_QUEUE_PROFILING_ENABLE, CL_TRUE, &old_priority_);
  }
  cl::Event *GetEvent(void) { return &event_; }
  
  ~Profiling() {
    profiling_status_ |= clSetCommandQueueProperty(queue_(),
        CL_QUEUE_PROFILING_ENABLE, CL_FALSE, NULL);
    profiling_status_ |= clSetCommandQueueProperty(queue_(), old_priority_,
        CL_TRUE, NULL);
    if (profiling_status_ != CL_SUCCESS) {
      std::cerr << "[Profile] " << id_ <<
        " failed restoring command queue property ("
        << profiling_status_<< ")"<<std::endl;
    }
  }
  void Report(size_t amount_s) {
    profiling_status_ |= queue_.finish();
    cl_ulong start_time, end_time;
    profiling_status_ |= clGetEventProfilingInfo(event_(),
        CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    profiling_status_ |= clGetEventProfilingInfo(event_(),
        CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    if (profiling_status_ == CL_SUCCESS) {
      std::cerr << "[Profile] " << id_ << " "<< end_time - start_time
          << "ns"; 
      if (amount_s) {
        double amount = amount_s; 
        double throughput = (amount)/(end_time-start_time);
        std::cerr << "; " << throughput << " GB/s" << std::endl;
      } else {
        std::cerr << std::endl;
      }
    } else {
      std::cerr << "[Profile] " << id_ << " profiling failed: " <<
        profiling_status_ << std::endl;
    }
  }
 private:
  std::string id_;
  cl_int profiling_status_;
  cl::CommandQueue queue_;
  cl::Event event_;
  cl_command_queue_properties old_priority_;
};
}

#endif // LIBMARSHAL_CL_PROFILE_H_
