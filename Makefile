CC = mpicc

NIXGL := $(shell command -v nixGL)

all: p4est

p4est:
	tar xzf external/p4est-*.tar.gz && mv p4est-* p4est
	cd p4est && ./configure CC=$(CC) --enable-debug --enable-mpi && $(MAKE) -j 4 install

setup:
	$(NIXGL) julia --project=. -e 'using Pkg; Pkg.add("Preferences")'
	$(NIXGL) julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
	$(NIXGL) julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
	$(NIXGL) julia --project=. -e 'using Pkg; Pkg.add("MPI"); Pkg.add("P4est")'
	$(NIXGL) julia --project=. -e 'using P4est; P4est.set_library_p4est!("p4est/local/lib/libp4est.so")'
	$(NIXGL) julia --project=. -e 'using Pkg; Pkg.add("P4estTypes"); Pkg.add("CUDA");'
	# $(NIXGL) julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"12.2", local_toolkit=true)'
	$(NIXGL) julia --project=. -e 'using CUDA; CUDA.versioninfo()'
	$(NIXGL) julia --project=. -e 'using Preferences; using CUDA; Preferences.set_preferences!(CUDA, "nonblocking_synchronization" => false)'
	$(NIXGL) julia --project=. -e 'using CUDA; @assert(CUDA.use_nonblocking_synchronization == false)'
	$(NIXGL) julia --project=. -e 'using Pkg; Pkg.add(name="Raven");'

# Rules
.PHONY: clean setup

clean:
	rm -rf p4est

realclean: clean
	git clean -X -d -f
