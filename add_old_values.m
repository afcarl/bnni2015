
%import the csv file manually!!!

addpath(genpath('DeepLearnToolbox'));
	addpath(genpath('Whetlab-Matlab-Client'));

	parameters = {struct('name', 'LR', 'type', 'float', 'min', 1e-6, 'max', 2.0, 'size', 1),...
		      struct('name', 'drop', 'type', 'float', 'min', 0.0, 'max', 0.9, 'size', 1),...
              struct('name', 'epochs', 'type', 'int', 'min', 10, 'max', 200, 'size', 1),...
		      struct('name', 'activation', 'type', 'enum', 'options', {{'tanh_opt' 'sigm' 'relu'}}, 'size', 1),...
		      struct('name', 'hidden', 'type', 'int', 'min', 32, 'max', 2048, 'size', 1),...
		      struct('name', 'momentum', 'type', 'float', 'min', 0.0, 'max', 0.99, 'size', 1),...
		      struct('name', 'LR_decrease', 'type', 'float', 'min', 0.75, 'max', 1.0, 'size', 1),...
		      struct('name', 'regularization', 'type', 'float', 'min', 1e-6, 'max', 0.5, 'size', 1)};
		  
	outcome.name = 'Accuracy';

		  
	scientist = whetlab('Whetlab_search_rat_epochs',...
		            'Try to find the rat!',...
		            parameters,...
		            outcome, true, '871cee3e-c6b4-460c-bd30-3ce9a48838ce');


nr_of_jobs=size(untitled,1)
for i=2:42
job.LR=untitled{i,3};
job.LR_decrease=untitled{i,4};
accuracy=untitled{i,2};
job.activation=untitled{i,5};
job.drop=untitled{i,6};
job.hidden=str2num(untitled{i,7}(5:end));
job.momentum=untitled{i,8};
job.regularization=untitled{i,9};
job.epochs=untitled{i,10};
scientist.update(job, accuracy)
end
