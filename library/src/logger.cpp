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

#include "include/logger.hpp"

#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mutex>

namespace hiptensor {
Logger::Logger()
    : mEnabled(true), mOwnsStream(false), mLogMask(0), mWriteStream(stdout),
      mCallback(nullptr) {}

Logger::~Logger() {
  std::scoped_lock lock(mMutex);
  if (mOwnsStream && mWriteStream != nullptr) {
    fclose(mWriteStream);
  }
}

Logger::Status_t Logger::writeToStream(FILE *stream) {
  std::scoped_lock lock(mMutex);
  if (stream != nullptr) {
    if (mOwnsStream && mWriteStream != nullptr) {
      fclose(mWriteStream);
      mOwnsStream = false;
    }
    mWriteStream = stream;
    return LOGGER_SUCCESS;
  }

  return LOGGER_BAD_FILE_STREAM;
}

Logger::Status_t Logger::openFileStream(const char *fileName) {
  std::scoped_lock lock(mMutex);
  if (fileName != nullptr && !strcmp(fileName, "")) {
    if (mOwnsStream && mWriteStream != nullptr) {
      fclose(mWriteStream);
      mOwnsStream = false;
    }
    mWriteStream = fopen(fileName, "w");
    if (mWriteStream == nullptr) {
      return LOGGER_FILE_OPEN_FAILED;
    }
    mOwnsStream = true;
    return LOGGER_SUCCESS;
  }

  // Revert back to stdout
  mOwnsStream = false;
  mWriteStream = stdout;
  return LOGGER_BAD_FILE_NAME;
}

Logger::Status_t Logger::setCallback(Callback_t callbackFunc) {
  std::scoped_lock lock(mMutex);
  if (callbackFunc != nullptr) {
    mCallback = callbackFunc;
    return LOGGER_SUCCESS;
  }
  return LOGGER_BAD_CALLBACK;
}

int32_t Logger::getLogMask() const {
  std::scoped_lock lock(mMutex);
  return mLogMask;
}

Logger::Status_t Logger::setLogMask(int32_t mask) {
  std::scoped_lock lock(mMutex);
  if (mask >= 0) {
    mLogMask = mask;
    return LOGGER_SUCCESS;
  }
  return LOGGER_BAD_LOG_MASK;
}

void Logger::disable() {
  std::scoped_lock lock(mMutex);
  mEnabled = false;
}

void Logger::enable() {
  std::scoped_lock lock(mMutex);
  mEnabled = true;
}

Logger::Status_t Logger::logMessage(int32_t context, const char *apiFuncName,
                                    const char *message) {
  std::scoped_lock lock(mMutex);
  if ((context & mLogMask) > 0 && mEnabled) {
    char buff[2048];
    sprintf(buff, "[%s][hipTensor][%d][%s][%s] %s", timeStamp(), appPid(),
            contextString(context), apiFuncName, message);

    if (mCallback != nullptr) {
      (*mCallback)(context, apiFuncName, buff);
    }

    fprintf(mWriteStream, "%s", buff);
  }
  return LOGGER_SUCCESS;
}

/* static */
const char *Logger::timeStamp() {
  static std::mutex tsMutex;
  std::scoped_lock lock(tsMutex);

  // Statically allocate buffer.
  // Beware thread concurrency.
  static char buff[32];

  time_t t;
  struct tm *tInfo;

  // Retrieve the time information
  time(&t);
  tInfo = localtime(&t);

  // Format the timestamp string
  // YYYY-MM-DD HH:MM:SS
  strftime(buff, 32, "%F %T", tInfo);
  return buff;
}

/* static */
int32_t Logger::appPid() {
  // App PID won't generally change
  static int pid = getpid();
  return pid;
}

/* static */
const char *Logger::contextString(int32_t context) {
  // NOTE: MUST align with hiptensorLogLevel_t
  switch (context) {
  case 1:
    return "Error";
  case 2:
    return "Performance";
  case 4:
    return "Kernel Selection";
  case 8:
    return "API";
  default:
    return "";
  }
  return "";
}

} // namespace hiptensor
