#pragma once

#ifndef QCOM_HTP_ARCH
#if defined(__HEXAGON_ARCH__)
#define QCOM_HTP_ARCH __HEXAGON_ARCH__
#elif defined(SM_MOE_DSP_ARCH)
#define QCOM_HTP_ARCH SM_MOE_DSP_ARCH
#else
#define QCOM_HTP_ARCH 79
#endif
#endif

#if QCOM_HTP_ARCH != 79 && QCOM_HTP_ARCH != 81 && QCOM_HTP_ARCH != 85
#error "Supported HTP architectures are V79, V81 and V85"
#endif
#if defined(__hexagon__) && QCOM_HTP_ARCH != __HEXAGON_ARCH__
#error "HTP architecture definition differs from the compiler target"
#endif

#define QCOM_HTP_V79 (QCOM_HTP_ARCH == 79)
#define QCOM_HTP_V81 (QCOM_HTP_ARCH == 81)
#define QCOM_HTP_V85 (QCOM_HTP_ARCH == 85)
