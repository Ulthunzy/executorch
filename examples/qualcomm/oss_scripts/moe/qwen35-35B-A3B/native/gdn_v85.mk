$(eval $(call HEXAGON_RULES,85))

$(WORK)/hexagon-v85:
	@mkdir -p $@/ops

htp_v85: $(WORK)/hexagon-v85/$(HTP_LIBRARY_NAME)

-include $(addprefix $(WORK)/hexagon-v85/,$(OTHER_OBJS:.o=.d) $(OP_OBJS:.o=.d))
.PHONY: htp_v85
