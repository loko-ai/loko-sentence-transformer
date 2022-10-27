from loko_extensions.model.components import Component, Input, Output, save_extensions, Select, Arg, Dynamic, \
    MultiKeyValue, MKVField

component_description = ""
fit_service = ""
predict_service = ""
evaluate_service = ""
# read_db_service = ""
create_service = "model/create"

model_name = Arg(name="model_name", label="Model Name", type="text", required=True,
                 helper="Name to assign to the model")

create_group = "Create Parameters"
pretrained_name = Arg(name="pretrained_name", label="Pretrained Model Name", type="text", required=True,
                      value="sentence-transformers/paraphrase-mpnet-base-v2", group=create_group)
is_multilabel = Arg(name="is_multilabel", label="Multilabel", type="boolean", value=False, group=create_group)
multi_target_strategy = Dynamic(name="multi_target_strategy", label="Multi-Target Strategy", dynamicType="select",
                                options=["one-vs-rest", "multi-output", "classifier-chain"], parent="is_multilabel", condition="{parent}===true", group=create_group)

create_args = [model_name, pretrained_name, is_multilabel, multi_target_strategy]
################################# SAVE ARGS ##################################
fit_group = "Fit Parameters"


#train_dataset, eval_dataset, loss, metric="accuracy", batch_size=16, n_iter=20, n_epochs=1, column_mapping=None

loss_val = ["CosineSimilarityLoss", "SoftmaxLoss", "MultipleNegativesRankingLoss", "MultipleNegativesSymmetricRankingLoss", "MSELoss", "MarginMSELoss","ContrastiveLoss", "ContrastiveTensionLoss", "DenoisingAutoEncoderLoss", "TripletLoss"]
loss_args = Select(name="loss", label="Loss function", options=loss_val, helper="Loss function to use, choose between the one of sentence_transformer Python library", value="CosineSimilarityLoss", group=fit_group)

metric = Arg(name="metric", label="Metric", type="text", value="accuracy", group=fit_group)

batch_size = Arg(name="batch_size", label="Batch Size", type="number", group=fit_group)

n_iter = Arg(name="n_iter", label="Number of iteration", type="number", group=fit_group)
n_epochs = Arg(name="n_epochs", label="Number of epochs", type="number", group=fit_group)

text_feature = Arg(name="text_feature", label="Textual feature name", type="text", group=fit_group)
label_feature = Arg(name="label_feature", label="Label feature name", type="text", group=fit_group)

# bucket_save = Arg(name="bucket_save", label="Bucket Name", type="text", group=save_group, value='influx-bu')
# measurement_name = Arg(name="measurement_name", label="Measurement Name", type="text", group=save_group)
#
# time_key = Arg(name="time", label="Time Key", type="text", group=save_group)


#
# mkvfields_tags = [MKVField(name="tag_key", label="Tag Key")]
# tags = MultiKeyValue(name="tags", label="Tags", fields=mkvfields_tags, group=save_group)
#
# mkvfields_fields = [MKVField(name="field_key", label="Field Key")]
# fields = MultiKeyValue(name="fields", label="Fields", fields=mkvfields_fields, group=save_group)
#
#
fit_args = [loss_args, metric, batch_size, n_iter, n_epochs, text_feature, label_feature]

################################# Delete ARGS #################################
predict_group = "Predict Parameters"

# bucket_del = Arg(name="bucket_del", label="Bucket Name", type="text", group=delete_group, value='influx-bu')
#
# from_query_del = Arg(name='from_query_del', label='From query', type='boolean', group=delete_group, value=False)
# measurement_del = Dynamic(name="measurement_delete", label="Measurement Name", parent='from_query_del',
#                           dynamicType="text", condition='!{parent}', group=delete_group)
#
#
# start_del = Arg(name="start_delete", label="Start", type="text",
#                 helper="Define the starting time to consider for deleting your data",
#                 group=delete_group, value="1970-01-01T00:00:00Z")
#
# stop_del = Arg(name="stop_delete", label="Stop", type="text",
#                helper= "Define the stopping time to consider for deleting your data",
#                group=delete_group)

predict_args = []

############################## READ ARGS ######################################
evaluate_group = "Evaluate Parameters"
#
# bucket_read = Arg(name="bucket_read", label="Bucket Name", type="text", group=read_group, value='influx-bu')
#
# start_read = Arg(name="start_read", label="Start", type="text",
#                   helper="Define the starting time to consider for reading your data",
#                   group=read_group,
#                   value="1970-01-01T00:00:00Z")
# stop_read = Arg(name="stop_read", label="Stop", type="text",
#                   helper="Define the stopping time to consider for reading your data",
#                   group=read_group)


evaluate_args = []

############# ARGS
args_list = create_args + fit_args + predict_args + evaluate_args

###############################################################################
input_list = [Input(id="create", label="Create Model", to="create", service=create_service),
              Input(id="fit", label="Fit", to="fit", service=fit_service),
              Input(id="predict", label="Predict", to="predict", service=predict_service),
              Input(id="evaluate", label="Evaluate", to="evaluate", service=evaluate_service),
              ]
output_list = [Output(id="create", label="Create Model"),
               # Output(id="fit", label="Fit"), Output(id="predict", label="Predict"), Output(id="evaluate", label="Evaluate")

               ]

sentence_transf_component = Component(name="SentenceTransformer", description=component_description, inputs=input_list,
                                      outputs=output_list, args=args_list, icon="RiTreasureMapFill", group='NLP')
# "RiFileTextFill"

save_extensions([sentence_transf_component])
