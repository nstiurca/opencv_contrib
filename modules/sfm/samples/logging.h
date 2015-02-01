/**
 * logging.h
 *
 *  Created on: Jul 5, 2012
 *    Modified: Jul 2014
 *      Author: Nicu Stiurca
 */
#pragma once

#include <cstdlib>
#include <iostream>

template<class T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &iter)
{
  typename std::vector<T>::const_iterator it = iter.begin();
  typename std::vector<T>::const_iterator end = iter.end();

  if(it == end) return out << '[' << ']'; //"[]";

  out << '[' << *it++;

  while(it != end) {
    out << ',' << ' ' << /*", " <<*/ *it++;
  }

  return out << ']';
}

#define HAVE_EXCEPTIONS 1
#if HAVE_EXCEPTIONS
#define MSG_BASE(stream, body) \
  do { try { \
    stream << __FILE__ << ":" << __FUNCTION__ << "():" << __LINE__ << ": " \
            << body << ::std::endl; \
  } catch (...) { /* don't throw anything while trying to output messages*/ } \
  } while(0)
#else // !HAVE_EXCEPTIONS
#define MSG_BASE(stream, body) \
  do { \
    stream << __FILE__ << ":" << __FUNCTION__ << "():" << __LINE__ << ": " \
           << body << ::std::endl; \
  } while(0)
#endif // HAVE_EXCEPTIONS

// DEBUG_STR(body) is conditionally defined
#define INFO_STR(body) MSG_BASE(::std::cout, "INFO:" << body)
#define WARN_STR(body) MSG_BASE(::std::cerr, "WARN:" << body)
#define ERROR_STR(body) MSG_BASE(::std::cerr, "ERROR:" << body)
#define FATAL_STR(body) MSG_BASE(::std::cerr, "FATAL:" << body)


#if defined(NDEBUG) || defined(NDEBUG_LOG)
#define DEBUG_STR(body)
#else
#define DEBUG_STR(body) MSG_BASE(::std::clog, "DEBUG:" << body)
#endif

#define NV(var) "  " << #var ": " << var

#define DEBUG(var) DEBUG_STR(NV(var))
#define INFO(var) INFO_STR(NV(var))
#define WARN(var) WARN_STR(NV(var))
#define ERROR(var) ERROR_STR(NV(var))
#define FATAL(var) FATAL_STR(NV(var))

