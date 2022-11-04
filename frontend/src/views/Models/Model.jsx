import { Box, HStack, Text, IconButton, Spacer, Stack, Tag, Button } from "@chakra-ui/react";
import { RiDeleteBin4Line, RiFolderDownloadLine } from "react-icons/ri";
import { useCompositeState } from "ds4biz-core";
import { useEffect } from "react";
import { Row, RowBody } from '../../utils/Row';

import { CLIENT } from "../../config/constants";


export function Model({ name, onDelete, onExport, ...rest }) {


  const state = useCompositeState({
    model: null,
    view: "general",
    pretrained_name: "Auto",
    is_multilabel: false,
    multi_target_strategy: null,
    status_tag: null
  });
  


  useEffect(() => {
    CLIENT.models[name]
    .get()
    .then((resp) => {state.model=resp.data
                      state.pretrained_name = resp.data.pretrained_name
                      state.status_tag = resp.data.fitted ??= false
                      state.is_multilabel = resp.data.is_multilabel ??= false
                      state.multi_target_strategy = resp.data.multi_target_strategy ??= null
                    })
    .catch((err) => console.log(err));
  }, []);


  let color_status = "#F9A602"
  let fit_status = "Not Fitted" 
  console.log("status:::", state.status_tag)
  if (state.status_tag===true) {
    color_status = "#29AB87";
    fit_status = "Fitted"
  } else if (state.status_tag=="fitting") {
    color_status = "#CA3433";
    fit_status = "Fitting"
    //#d06464
  } 
  console.log("status updated::: ", fit_status)

  let multilabel_tag = "False"
  if (state.is_multilabel){
    multilabel_tag = "True"
  }

  let multi_target_strategy_tag = "-"
  if (state.multi_target_strategy !== null){
    multi_target_strategy_tag = state.multi_target_strategy
  }

  return (
    <HStack
      bg="gray.200"
      borderRadius={"10px"}
      w="100%"
      py="0.5rem"
      px="1rem"
      {...rest}
    >
      <Stack spacing={0}>
        <HStack color={"#81007F"}>
            <Box><Text as="b" color='#81007F"'>{name}</Text></Box>
            <Tag borderRadius={"10px"} p=".3rem" bg={color_status} color="white" fontSize="xs">
              <b>{fit_status}</b>
            </Tag>
        </HStack>
        <RowBody>
        {state.model?.description && (
                              <Text
                                alignSelf="flex-start"
                                mb="4"
                                as="h3"
                                size=""
                                color="#808089"
                                fontWeight="500"
                                wordBreak="break-all"
                              >
                                {state.model?.description}
                              </Text>
                            )}
        </RowBody>
        <HStack fontSize={"xs"}>
            <Stack spacing="0">
              <Box><Text as="b">Pre-Trained Model</Text></Box>
              <Box>{state.pretrained_name}</Box>
            </Stack>
            <Stack spacing="0">
              <Box><Text as="b">Multilabel</Text></Box>
              <Box>{multilabel_tag}</Box>
            </Stack>
            <Stack spacing="0">
              <Box><Text as="b">Multitarget Strategy</Text></Box>
              <Box>{multi_target_strategy_tag}</Box>
            </Stack>
          </HStack>
      </Stack>
      <Spacer />
      <HStack>
        <IconButton
          size="sm"
          borderRadius={"full"}
          icon={<RiDeleteBin4Line />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onDelete(e);
          }}
        />
        <IconButton
          size="sm"
          borderRadius={"full"}
          icon={<RiFolderDownloadLine />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onExport(e);
          }}
        />
      </HStack>
    </HStack>
  );
}
