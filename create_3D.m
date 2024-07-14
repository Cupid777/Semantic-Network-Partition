file = dir('/feature_selection_new/mat_input_4d/*.mat');
nii_data = load_nii('/feature_selection_new/7_network_semantic_ridge.nii')
meta = createMetaFromMask(nii_data.img,'radius',1);
save(['/feature_selection_new/7_network_semantic_ridge.mat'],'meta');
for j=1:length(file)
    load(['/feature_selection_new/mat_input_4d/' file(j).name]);
    examples = createExamplesFrom4D(mid,meta);
    save(['mat_input_3D_semantic_ridge' '/' file(j).name],'examples');
end

file = dir('/feature_selection_new/mat_input_4d/*.mat');
nii_data = load_nii('/eature_selection_new/7_network_semantic_RSA.nii')
meta = createMetaFromMask(nii_data.img,'radius',1);
save(['/feature_selection_new/7_network_semantic_RSA.mat'],'meta');
for j=1:length(file)
    load(['/feature_selection_new/mat_input_4d/' file(j).name]);
    examples = createExamplesFrom4D(mid,meta);
    save(['mat_input_3D_semantic_RSA' '/' file(j).name],'examples');
end