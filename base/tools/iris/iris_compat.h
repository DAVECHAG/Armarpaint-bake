/*
 * iris_compat.h - Windows portability shims
 *
 * Iris is written against POSIX. This header provides the small subset of
 * POSIX functionality the sources actually use (memory-mapped files, a
 * monotonic-ish wall clock, case-insensitive compares, and getopt_long) so the
 * engine builds with the MSVC/clang toolchain on Windows. On non-Windows
 * platforms this header is empty and the normal system headers are used.
 *
 * Include this BEFORE the POSIX headers it replaces. The build guards those
 * includes (<sys/mman.h>, <unistd.h>, <sys/time.h>, <dirent.h>, <getopt.h>)
 * with #ifndef _WIN32, and includes this header in the #else branch.
 */
#ifndef IRIS_COMPAT_H
#define IRIS_COMPAT_H

#ifdef _WIN32

/* Quiet the deprecation warnings on the POSIX CRT names (open/close/strdup/...)
 * and keep <windows.h> from dragging in winsock + min/max macros. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_WARNINGS
#define _CRT_NONSTDC_NO_WARNINGS
#endif

#include <windows.h>

#include <fcntl.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* --- <strings.h> -------------------------------------------------------- */
#define strcasecmp  _stricmp
#define strncasecmp _strnicmp

/* --- 64-bit stat -------------------------------------------------------- */
/* The default Windows `struct stat` stores st_size in a 32-bit long, which
 * truncates the multi-GB model files. Route stat()/fstat() through the 64-bit
 * variants (struct _stat64 / _stat64 / _fstat64 share the same spelling). */
#define stat  _stat64
#define fstat _fstat64

/* --- <sys/mman.h> (read-only file mappings) ----------------------------- */
#define PROT_READ   0x1
#define PROT_WRITE  0x2
#define MAP_SHARED  0x1
#define MAP_PRIVATE 0x2
#define MAP_FAILED  ((void *)-1)

/* Map a CRT file descriptor. offset is 0 in every iris call site. The view
 * stays valid after the mapping object is closed, so munmap only unmaps. */
static __inline void *mmap(void *addr, size_t length, int prot, int flags, int fd, long long offset) {
	(void)addr;
	(void)prot;
	(void)flags;
	HANDLE fh = (HANDLE)_get_osfhandle(fd);
	if (fh == INVALID_HANDLE_VALUE)
		return MAP_FAILED;
	HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
	if (!mh)
		return MAP_FAILED;
	void *p = MapViewOfFile(mh, FILE_MAP_READ, (DWORD)((unsigned long long)offset >> 32), (DWORD)((unsigned long long)offset & 0xFFFFFFFFu), length);
	CloseHandle(mh);
	return p ? p : MAP_FAILED;
}

static __inline int munmap(void *addr, size_t length) {
	(void)length;
	return UnmapViewOfFile(addr) ? 0 : -1;
}

/* --- <sys/time.h> ------------------------------------------------------- */
/* WIN32_LEAN_AND_MEAN keeps winsock (and its struct timeval) out, so we own
 * the definition here. */
struct timeval {
	long tv_sec;
	long tv_usec;
};

static __inline int gettimeofday(struct timeval *tv, void *tz) {
	(void)tz;
	FILETIME       ft;
	ULARGE_INTEGER t;
	GetSystemTimeAsFileTime(&ft);
	t.LowPart  = ft.dwLowDateTime;
	t.HighPart = ft.dwHighDateTime;
	/* 100ns ticks since 1601-01-01 -> microseconds since the Unix epoch. */
	t.QuadPart -= 116444736000000000ULL;
	tv->tv_sec  = (long)(t.QuadPart / 10000000ULL);
	tv->tv_usec = (long)((t.QuadPart % 10000000ULL) / 10ULL);
	return 0;
}

/* --- <unistd.h>: sysconf(_SC_NPROCESSORS_ONLN) -------------------------- */
#define _SC_NPROCESSORS_ONLN 1

static __inline long sysconf(int name) {
	if (name == _SC_NPROCESSORS_ONLN) {
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		return (long)si.dwNumberOfProcessors;
	}
	return -1;
}

/* --- <getopt.h> --------------------------------------------------------- */
/* Compiled into the single translation unit (main.c) that defines
 * IRIS_NEED_GETOPT before including this header. */
#ifdef IRIS_NEED_GETOPT

#define no_argument       0
#define required_argument 1
#define optional_argument 2

struct option {
	const char *name;
	int         has_arg;
	int        *flag;
	int         val;
};

char *optarg = NULL;
int   optind = 1;
int   opterr = 1;
int   optopt = 0;

static int iris_optpos = 1; /* position within a clustered short-option group */

static int getopt_long(int argc, char *const argv[], const char *optstring, const struct option *longopts, int *longindex) {
	optarg = NULL;
	if (optind >= argc)
		return -1;

	char *arg = argv[optind];
	if (arg[0] != '-' || arg[1] == '\0')
		return -1; /* not an option; stop (no permutation) */

	if (arg[1] == '-') {
		/* "--" terminator or a long option */
		if (arg[2] == '\0') {
			optind++;
			return -1;
		}
		const char *name    = arg + 2;
		const char *eq      = strchr(name, '=');
		size_t      namelen = eq ? (size_t)(eq - name) : strlen(name);
		for (int i = 0; longopts && longopts[i].name; i++) {
			if (strlen(longopts[i].name) == namelen && strncmp(name, longopts[i].name, namelen) == 0) {
				optind++;
				if (longindex)
					*longindex = i;
				if (longopts[i].has_arg == required_argument) {
					if (eq)
						optarg = (char *)eq + 1;
					else if (optind < argc)
						optarg = argv[optind++];
					else {
						if (opterr)
							fprintf(stderr, "%s: option '--%s' requires an argument\n", argv[0], longopts[i].name);
						return '?';
					}
				}
				else if (longopts[i].has_arg == optional_argument && eq) {
					optarg = (char *)eq + 1;
				}
				if (longopts[i].flag) {
					*longopts[i].flag = longopts[i].val;
					return 0;
				}
				return longopts[i].val;
			}
		}
		if (opterr)
			fprintf(stderr, "%s: unrecognized option '--%s'\n", argv[0], name);
		optind++;
		return '?';
	}

	/* short option (possibly clustered, e.g. -qv) */
	int         c = arg[iris_optpos];
	const char *p = (c != ':') ? strchr(optstring, c) : NULL;
	if (!p) {
		optopt = c;
		if (opterr)
			fprintf(stderr, "%s: invalid option -- '%c'\n", argv[0], c);
		if (arg[++iris_optpos] == '\0') {
			optind++;
			iris_optpos = 1;
		}
		return '?';
	}

	if (p[1] == ':') {
		/* option takes an argument */
		if (arg[iris_optpos + 1] != '\0') {
			optarg = &arg[iris_optpos + 1];
			optind++;
		}
		else if (optind + 1 < argc) {
			optarg = argv[optind + 1];
			optind += 2;
		}
		else {
			optopt = c;
			optind++;
			iris_optpos = 1;
			if (opterr)
				fprintf(stderr, "%s: option requires an argument -- '%c'\n", argv[0], c);
			return '?';
		}
		iris_optpos = 1;
		return c;
	}

	/* option without an argument */
	if (arg[++iris_optpos] == '\0') {
		optind++;
		iris_optpos = 1;
	}
	return c;
}

#endif /* IRIS_NEED_GETOPT */

#endif /* _WIN32 */
#endif /* IRIS_COMPAT_H */
