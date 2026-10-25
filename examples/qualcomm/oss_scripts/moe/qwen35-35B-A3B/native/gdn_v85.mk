# Makefile has already expanded HEXAGON_CPPFLAGS into the base flags here.
HEXAGON_BASE_FLAGS += -DDELTANET_U16_KARATSUBA_REDUCTION=1 -O3

$(eval $(call HEXAGON_RULES,85))

$(WORK)/hexagon-v85:
	@mkdir -p $@/ops

htp_v85: $(WORK)/hexagon-v85/$(HTP_LIBRARY_NAME)

-include $(addprefix $(WORK)/hexagon-v85/,$(OTHER_OBJS:.o=.d) $(OP_OBJS:.o=.d))
.PHONY: htp_v85
