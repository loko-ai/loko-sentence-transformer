import { Button, Flex, HStack, Stack } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useContext, useRef} from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { Model } from "./Model";
import { ModelCreation } from "./ModelCreation";
import { RiAddFill, RiUploadCloud2Line } from 'react-icons/ri';
import { ModelDetails } from "./ModelDetails";
import { saveAs } from 'file-saver';



export function Models({ models }) {
  const state = useCompositeState({ view: "list" });
  const _state = useContext(StateContext);
  const ref_import = useRef();

  switch (state.view) {
    case "list":
      return (
        <Stack w="100%" h="100%" spacing="2rem">
          <HStack>
            <Button onClick={(e) => (state.view = "new")} leftIcon={<RiAddFill />}>New model</Button>
            <Button leftIcon={<RiUploadCloud2Line />}
                onClick={(e) => {
                    console.log('click');
                    ref_import.current.click()
                }}
            >
             Import
             </Button>
             <input
                type='file'
                accept=".zip"
                ref={ref_import}
                onChange={(e) => {
                    console.log('change import');
                    console.log(e.target.files[0]);
                    const formData = new FormData();
                    formData.append('f', e.target.files[0]);
                    CLIENT.models.import.post(formData).then(()=>location.reload()).catch((err) => console.log(err));
                }}
                onSubmit={(e) => {
                    e.preventDefault();
                    console.log('submit import');
                    _state.refresh = new Date();
                    // location.reload();
                }}
                style={{ display: 'none' }}/>
          </HStack>

          <Stack>
            {models.map((name) => (
              <Model
                onClick={(e) => {state.view = "show_blueprint", state.name ={name}}}
                name={name}
                key={name}
                onDelete={(e) =>
                  CLIENT.models[name].delete().then((resp) => {
                    _state.refresh = new Date();
                  })
                }
                onExport={(e) =>
                  CLIENT.models[name].export.get({responseType: "arraybuffer"})
                  .then(response => {
                    console.log('download');
                    console.log(response);
                    const blob = new Blob([response.data], {
                            type: 'application/octet-stream'
                            })
                    return blob
                    })
                    .then(blob => {
                    console.log(blob)
                    const filename = name+'.zip'
                    saveAs(blob, filename)
                    console.log('hello');
                    })
                  .catch(error => {
                    console.log(error);
                    })
                   }
              />
            ))}
          </Stack>
        </Stack>
      );
    case "new":
      return (
        <Flex w="100vw" h="100vh" p="2rem">
          <ModelCreation onClose={(e) => (state.view = "list")} />
        </Flex>
      );
    case "show_blueprint":
      console.log("Models in detailssssssss::::")
      return (
        <Flex w="100vw" h="100vh" p="2rem">
          <ModelDetails onClose={(e) => (state.view = "list")} name={Object.values(state.name)} />
        </Flex>

      );
      

    default:
      break;
  }
}
