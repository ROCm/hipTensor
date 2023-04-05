/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include <fstream>
#include <hip/hip_runtime.h>

inline void hip_check_error(hipError_t x) {
  if (x != hipSuccess) {
    std::ostringstream ss;
    ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__
       << ": " << __LINE__ << "in function: " << __func__;
    throw std::runtime_error(ss.str());
  }
}

template <typename T> void hiptensorPrintArrayElements(T *vec, size_t size) {
  int index = 0;
  while (index != size) {
    if (index == size - 1)
      std::cout << vec[index];
    else
      std::cout << vec[index] << ",";

    index++;
  }
}

template <typename S>
void hiptensorPrintVectorElements(const std::vector<S> &vec,
                                  std::string sep = " ") {
  for (auto elem : vec) {
    std::cout << elem << sep;
  }
}

template <typename F>
void hiptensorPrintElementsToFile(std::ofstream &fs, F *output, size_t size,
                                  char delim) {
  if (!fs.is_open()) {
    std::cout << "File not found!\n";
    return;
  }

  for (int i = 0; i < size; i++) {
    if (i == size - 1)
      fs << static_cast<F>(output[i]);
    else
      fs << static_cast<F>(output[i]) << delim;
  }
  return;
}
