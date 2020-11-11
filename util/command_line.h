/* 
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 NVIDIA
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once
#include "common_header.h"

namespace cublas {
class CommandLine {

protected:
  std::map<std::string, std::pair<std::string, bool>> pairs;

public:
  CommandLine(int argc, char **argv) { add_args(argc, argv); }

  void add_args(int argc, char **argv) {
    using namespace std;

    for (int i = 1; i < argc; i++) {

      string arg = argv[i];

      if (arg[0] != '-') {
        continue;
      }

      string::size_type pos;
      string key, val;
      if ((pos = arg.find('=')) == string::npos) {
        key = string(arg, 1, arg.length() - 1);
        val = "";
      } else {
        key = string(arg, 1, pos - 1);
        val = string(arg, pos + 1, arg.length() - 1);
      }

      pairs[key] = std::make_pair(val, false);
    }
  }

  bool check_cmd_line_flag(const char *arg_name) {
    using namespace std;
    if (pairs.find(arg_name) != pairs.end()) {
      return true;
    }
    return false;
  }

  template <typename T>
  bool get_cmd_line_argument(const char *arg_name, T &val);
  int ParsedArgc() { return (int) pairs.size(); }

  bool all_flags_checked() {
    bool total = true;
    for (auto const &flag : pairs) {
      if (!flag.second.second) {
        total = false;
        printf("unknown argument %s=%s\n", flag.first.c_str(), flag.second.first.c_str());
      }
    }
    return total;
  }
};

template <typename T>
inline bool CommandLine::get_cmd_line_argument(const char *arg_name, T &val) {
  using namespace std;
  auto itr = pairs.find(arg_name);
  if (itr != pairs.end()) {
    itr->second.second = true;
    istringstream strstream(itr->second.first);
    strstream >> val;
    return true;
  }
  return false;
}

template <>
inline bool CommandLine::get_cmd_line_argument<char *>(const char *arg_name,
                                                       char *&val) {
  using namespace std;
  auto itr = pairs.find(arg_name);
  if (itr != pairs.end()) {

    itr->second.second = true;
    string s = itr->second.first;
    val = (char *)malloc(sizeof(char) * (s.length() + 1));
    std::strcpy(val, s.c_str());
    return true;
  } else {
    val = NULL;
  }
  return false;
}
} // namespace cublas
