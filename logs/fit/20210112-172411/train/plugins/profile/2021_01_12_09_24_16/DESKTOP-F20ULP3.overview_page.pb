?$	2U0*????lWX??????(????!d?]K???	j??O75@y?3??U@!j??O75!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$d?]K?????ͪ?զ?AHP?s??Y&S??:??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??(???????S㥛?A?+e?Xw?*	??????s@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap<?R?!???!O??)x?J@)??W?2???1?r
^N!F@:Preprocessing2F
Iterator::ModelˡE?????!9/??%*@)e?X???14և??&@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?;Nё\??!և????2@) o?ŏ??1     ?%@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????????!Y?C?@)Zd;?O???1???>4V@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeate?X???!4և??&@)?e??a???1????S0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????????!Y?C?@)????????1Y?C?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???<,Ԋ?!~h???@)?(??0??1x9/?`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipvOjM??!c}h?Q@)?0?*??1?>4ևF
@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchlxz?,C|?!???X?@)lxz?,C|?1???X?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice_?Q?{?!և???X@)_?Q?{?1և???X@:Preprocessing2U
Iterator::Model::ParallelMapV2-C??6z?!????S @)-C??6z?1????S @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatea2U0*???!?Cc}@)Ǻ???f?1$I?$I???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mb`?!?Cc}h??)????Mb`?1?Cc}h??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor-C??6J?!????S??)-C??6J?1????S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s8.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9"?&?q? @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??d?`T??q?eF?|?????S㥛?!??ͪ?զ?	!       "	!       *	!       2$	o??ʡ??h.?????+e?Xw?!HP?s??:	!       B	!       J	&S??:????K?}]??!&S??:??R	!       Z	&S??:????K?}]??!&S??:??JCPU_ONLYY"?&?q? @b Y      Y@q~7<d?1@"?	
both?Your program is MODERATELY input-bound because 8.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s8.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?17.5432% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 