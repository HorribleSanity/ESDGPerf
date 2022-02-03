const add_contract = true
if add_contract
  import Base: +, *, -, /
  for (jlf, f) in zip((:+, :*, :-, :/), (:add, :mul, :sub, :div))
      for (T, llvmT) in ((Float32, "float"), (Float64, "double"))
          ir = """
              %x = f$f contract nsz $llvmT %0, %1
              ret $llvmT %x
          """
          @eval begin
              # the @pure is necessary so that we can constant propagate.
              Base.@pure function $jlf(a::$T, b::$T)
                  Base.@_inline_meta
                  Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
              end
          end
      end
      @eval function $jlf(args...)
          Base.$jlf(args...)
      end
  end
end
