#pragma once

#include <iostream>
#include <string>
#include <stdio.h>

#define DB_MISC		1U << 0
#define DB_SYNC		1U << 1
#define DB_AQL		1U << 2
#define DB_QUEUE	1U << 3

#define DEBUG_ENABLED 1

char* hcc_db;
std::ostream &err = std::cerr;

#define DEBUG_MESSAGE_PRINTF(category, format, ...) \
hcc_db = std::getenv("HCC_DB"); \
if (hcc_db != NULL && DEBUG_ENABLED) { \
	int db_bitmask = atoi(hcc_db); \
	if (db_bitmask & category) {  \
		fprintf (stderr, format, __VA_ARGS__); \
	} \
}

template <typename T>
inline void DEBUG_MESSAGE_CERR(T t) {
	err << t;
}

template <typename T, typename... Mssg>
inline void DEBUG_MESSAGE_CERR(T t, Mssg... mssg) {
	err << t;
	DEBUG_MESSAGE_CERR(mssg...);
}

template <typename T, typename... Mssg>
inline void DEBUG_MESSAGE(T t, Mssg... mssg) {
	int category = t;

	hcc_db = std::getenv("HCC_DB");
	if (hcc_db == NULL || !DEBUG_ENABLED)
		return;

	int db_bitmask = atoi(hcc_db);

	if (!(db_bitmask & category))
		return;

	DEBUG_MESSAGE_CERR(mssg...);

	return;
}
