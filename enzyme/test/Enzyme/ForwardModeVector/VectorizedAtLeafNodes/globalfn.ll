; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double, i64)*, ...) #2

@global = private unnamed_addr constant [1 x void (double*)*] [void (double*)* @ipmul]

@.str = private unnamed_addr constant [6 x i8] c"x=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"xp=%f\0A\00", align 1

define void @ipmul(double* %x) {
entry:
  %0 = load double, double* %x, !tbaa !2
  %mul = fmul fast double %0, %0
  store double %mul, double* %x
  ret void
}

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @mulglobal(double %x, i64 %idx) #0 {
entry:
  %alloc = alloca double
  store double %x, double* %alloc
  %arrayidx = getelementptr inbounds [1 x void (double*)*], [1 x void (double*)*]* @global, i64 0, i64 %idx
  %fp = load void (double*)*, void (double*)** %arrayidx
  call void %fp(double* %alloc)
  %ret = load double, double* %alloc, !tbaa !2
  ret double %ret
}

; Function Attrs: noinline nounwind uwtable
define dso_local <3 x double> @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call <3 x double> (double (double, i64)*, ...) @__enzyme_fwddiff(double (double, i64)* nonnull @mulglobal, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>, i64 0)
  ret <3 x double> %0
}

; Function Attrs: nounwind uwtable
define dso_local void @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #3 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, !tbaa !6
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #2
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), double %call.i)
  %call2 = tail call <3 x double> @derivative(double %call.i)
  %1 = extractelement <3 x double> %call2, i32 0
  %call3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), double %1)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}


; CHECK: define internal <3 x double> @fwddiffe3mulglobal(double %x, <3 x double> %"x'", i64 %idx)
; CHECK-NEXT:  entry:
; CHECK-NEXT:   %"alloc'ipa" = alloca <3 x double>, align 8
; CHECK-NEXT:   store <3 x double> zeroinitializer, <3 x double>* %"alloc'ipa", align 8
; CHECK-NEXT:   %alloc = alloca double, align 8
; CHECK-NEXT:   store double %x, double* %alloc, align 8
; CHECK-NEXT:   store <3 x double> %"x'", <3 x double>* %"alloc'ipa", align 8
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds [1 x <3 x void (double*)*>], [1 x <3 x void (double*)*>]* @global_shadow, i64 0, i64 %idx
; CHECK-NEXT:   %arrayidx = getelementptr inbounds [1 x void (double*)*], [1 x void (double*)*]* @global, i64 0, i64 %idx
; CHECK-NEXT:   %"fp'ipl" = load <3 x void (double*)*>, <3 x void (double*)*>* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %fp = load void (double*)*, void (double*)** %arrayidx, align 8
; CHECK-NEXT:   %0 = extractelement <3 x void (double*)*> %"fp'ipl", i32 0
; CHECK-NEXT:   %1 = bitcast void (double*)* %fp to i8*
; CHECK-NEXT:   %2 = bitcast void (double*)* %0 to i8*
; CHECK-NEXT:   %3 = icmp eq i8* %1, %2
; CHECK-NEXT:   br i1 %3, label %error.i, label %__enzyme_runtimeinactiveerr.exit

; CHECK: error.i:                                          ; preds = %entry
; CHECK-NEXT:   %4 = call i32 @puts(i8* getelementptr inbounds ([79 x i8], [79 x i8]* @.str.2, i32 0, i32 0))
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: __enzyme_runtimeinactiveerr.exit:                 ; preds = %entry
; CHECK-NEXT:   %5 = bitcast void (double*)* %0 to void (double*, <3 x double>*)**
; CHECK-NEXT:   %6 = load void (double*, <3 x double>*)*, void (double*, <3 x double>*)** %5, align 8
; CHECK-NEXT:   call void %6(double* %alloc, <3 x double>* %"alloc'ipa")
; CHECK-NEXT:   %"ret'ipl" = load <3 x double>, <3 x double>* %"alloc'ipa", align 8, !tbaa !3
; CHECK-NEXT:   ret <3 x double> %"ret'ipl"
; CHECK-NEXT: }


; CHECK: define internal void @fwddiffe3ipmul(double* %x, <3 x double>* %"x'")
; CHECK-NEXT:  entry:
; CHECK-NEXT:   %"'ipl" = load <3 x double>, <3 x double>* %"x'", align 8, !tbaa !3
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !3
; CHECK-NEXT:   %mul = fmul fast double %0, %0
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %0, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %0, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %1 = fmul fast <3 x double> %"'ipl", %.splat2
; CHECK-NEXT:   %2 = fmul fast <3 x double> %"'ipl", %.splat
; CHECK-NEXT:   %3 = fadd fast <3 x double> %1, %2
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   store <3 x double> %3, <3 x double>* %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }