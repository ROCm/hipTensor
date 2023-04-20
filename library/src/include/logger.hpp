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

#ifndef HIPTENSOR_LOGGER_HPP
#define HIPTENSOR_LOGGER_HPP

#include "singleton.hpp"

#include <mutex>

namespace hiptensor {
class Logger : public LazySingleton<Logger> {
private:
  using Callback_t = void (*)(int32_t logLevel, const char *funcName,
                              const char *msg);

public:
  enum Status_t : int32_t {
    LOGGER_SUCCESS = 0,
    LOGGER_BAD_FILE_NAME,
    LOGGER_BAD_FILE_STREAM,
    LOGGER_BAD_CALLBACK,
    LOGGER_BAD_LOG_MASK,
    LOGGER_FILE_OPEN_FAILED,
  };

  // For static initialization
  friend std::unique_ptr<Logger> std::make_unique<Logger>();

  Logger();
  ~Logger();

  Status_t writeToStream(FILE *stream);
  Status_t openFileStream(const char *fileName);
  Status_t setCallback(Callback_t callbackFunc);
  int32_t getLogMask() const;
  Status_t setLogMask(int32_t mask);
  void disable();
  void enable();

  Status_t logMessage(int32_t context, const char *apiFuncName,
                      const char *message);

private:
  static const char *timeStamp();
  static int32_t appPid();
  static const char *contextString(int32_t context);

private:
  bool mEnabled;
  bool mOwnsStream;

  int32_t mLogMask;
  FILE *mWriteStream;
  Callback_t mCallback;

  mutable std::mutex mMutex;
};

} // namespace hiptensor

#endif // HIPTENSOR_LOGGER_HPP
