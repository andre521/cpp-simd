include makeci/ci.mk makeci/cmake.mk

ci-pre: gitinit
ci-init:
ci-build: test
ci-finish: gitdeinit

test: test.target
test.target: simd_test.target

ifndef VERBOSE
.SILENT:
endif
