import { Box, Button, Input, Stack, Text, Textarea } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";

export function BaseForm({ onSubmit }) {
  const state = useCompositeState({
    name: "",
    description: "",
    pretrained_name:"sentence-transformers/paraphrase-mpnet-base-v2",
    is_multilabel: false,
    multi_target_strategy: ""
  });
  return (
    <Stack>
      <Text fontSize="xs">
        Name
        <Box as="span" color="red">
          *
        </Box>
      </Text>
      <Input
        value={state.name}
        onChange={(e) => (state.name = e.target.value)}
        type="text"
      />

      <Text fontSize="xs">
        Description
        {/* <Box as="span" color="red">
          *
        </Box> */}
      </Text>
      <Textarea
        rows={2}
        value={state.description}
        onChange={(e) => (state.description = e.target.value)}
      />
      <Text fontSize="xs">
        Pre-Trained Name
        <Box as="span" color="red">
          *
        </Box>
      </Text>
      <Input
        value={state.pretrained_name}
        onChange={(e) => (state.pretrained_name = e.target.value)}
        type="text"
      />
      <Text fontSize="xs">
        Multilabel
      </Text>
      <Input
        value={state.is_multilabel}
        onChange={(e) => (state.is_multilabel = e.target.value)}
        type="boolean"
      />
      <Text fontSize="xs">
        Multi-Target Strategy
      </Text>
      <Input
        value={state.multi_target_strategy}
        onChange={(e) => (state.multi_target_strategy = e.target.value)}
        type="text"
      />

      <Button onClick={(e) => onSubmit(state.name, state.bp)}>Create</Button>
    </Stack>
  );
}
