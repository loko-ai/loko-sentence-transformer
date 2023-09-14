import { Box, Button, Input, Select, Stack, Text, Textarea } from "@chakra-ui/react";
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
        isInvalid={state.name === ""}
      />
      <Text fontSize="xs">
        Description
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
      <Select
        // option={[{varlue:true, label:"true"},{varlue:false, label:"false"} ]}
        // value={state.model_id}
        onChange={(e) => (state.is_multilabel = e.target.value)}
        type="boolean"
        defaultValue={false}

      >
        <option value={true}>true</option>
        <option value={false}>false</option>
      </Select>
      {/* <Input
        value={state.is_multilabel}
        onChange={(e) => (state.is_multilabel = e.target.value)}
        type="boolean"
    
        // {<option key={true}>True</option>
        // <option key={false}>False</option>}
      /> */}
      <Text fontSize="xs">
        Multi-Target Strategy
      </Text>
      <Input
        value={state.multi_target_strategy}
        onChange={(e) => (state.multi_target_strategy = e.target.value)}
        type="text"
      />

      <Button onClick={(e) => onSubmit(state.name, state.description, state.pretrained_name, state.is_multilabel, state.multi_target_strategy)}>Create</Button>
    </Stack>
  );
}
