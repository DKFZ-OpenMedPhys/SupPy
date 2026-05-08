load('HEAD_AND_NECK.mat');
%%
%%

cst{15,6}{1} = DoseConstraints.matRad_MinMaxDose(61,65);


cst{16,6}{1} = DoseConstraints.matRad_MinMaxDose(68,72);
%cst{16,6}{3} = DoseConstraints.matRad_MinMaxDVH(67.5,95,100);

cst{19,6}{1} = DoseConstraints.matRad_MinMaxDose(0,40);
cst{19,6}{2} = DoseConstraints.matRad_MinMaxDVH(10,0,5);

cst{13,6}{1} = DoseConstraints.matRad_MinMaxDose(0,40);
cst{13,6}{2} = DoseConstraints.matRad_MinMaxDVH(10,0,5);

cst{14,6}{1} = DoseConstraints.matRad_MinMaxDose(0,40);
cst{14,6}{2} = DoseConstraints.matRad_MinMaxDVH(10,0,5);

cst{7,6}{1} = DoseConstraints.matRad_MinMaxDose(0,40);
cst{7,6}{2} = DoseConstraints.matRad_MinMaxDVH(10,0,5);


cst{2,6}{1} = DoseConstraints.matRad_MinMaxDose(0,30);
cst{2,6}{2} = DoseConstraints.matRad_MinMaxDVH(10,0,5);


cst{17,6}{1} = DoseConstraints.matRad_MinMaxDose(0,65);
cst{17,6}{2} = DoseObjectives.matRad_MeanDose(100,0);

% %% Optimization reference
% % use standard plan and add some objectives
%
% %Brain Stem
% cst{2,6}{1} = DoseObjectives.matRad_MaxDVH(100,10,5);
% cst{2,6}{2} = DoseObjectives.matRad_SquaredOverdosing(100,40);
%
% %Larynx
% cst{7,6}{1} = DoseObjectives.matRad_MaxDVH(100,10,5);
% cst{7,6}{2} = DoseObjectives.matRad_SquaredOverdosing(100,40);
%
% %Parotids
% cst{13,6}{1} = DoseObjectives.matRad_MaxDVH(100,10,5);
% cst{13,6}{2} = DoseObjectives.matRad_SquaredOverdosing(100,40);
% cst{14,6}{1} = DoseObjectives.matRad_MaxDVH(100,10,5);
% cst{14,6}{2} = DoseObjectives.matRad_SquaredOverdosing(100,40);
%
%
% %PTV stays
%
% %Skin
% %cst{17,6}{1} = DoseObjectives.matRad_SquaredOverdosing(100,
% %cst{17,6}{1} = DoseConstraints.matRad_MinMaxDose(0,70);
% %cst{17,6}{2} = DoseObjectives.matRad_MeanDose(100);
% cst{17,6} = [];
%
% %Spinal Cord
% cst{19,6}{1} = DoseObjectives.matRad_MaxDVH(100,10,5);
% cst{19,6}{2} = DoseObjectives.matRad_SquaredOverdosing(100,40);
%
% % cst{2,6}{1} = DoseObjectives.matRad_SquaredOverdosing(100,15);
%
% %cst{19,6}{1} = DoseObjectives.matRad_SquaredOverdosing(100,15);
%
% %cst{7,6}{1} = DoseObjectives.matRad_SquaredOverdosing(100,15);


%%
matRadGUI
%%
% First of all, we need to define what kind of radiation modality we would
% like to use. Possible values are photons, protons or carbon. In this
% example we would like to use protons for treatment planning. Next, we
% need to define a treatment machine to correctly load the corresponding
% base data. matRad features generic base data in the file
% 'proton_Generic.mat'; consequently the machine has to be set accordingly
pln.radiationMode   = 'photons';
pln.machine         = 'Generic';
pln.numOfFractions  = 30;

%%
% for particles it is possible to also calculate the LET disutribution
% alongside the physical dose. Therefore you need to activate the
% corresponding option during dose calculcation. We also explicitly say to
% use the Hong Pencil Beam Algorithm
pln.bioModel = 'none';
pln.multScen = 'nomScen';

%%
% Now we have to set the remaining plan parameters.
pln.propStf.gantryAngles   = [0:40:359];
pln.propStf.couchAngles    = zeros(1,numel(pln.propStf.gantryAngles));
pln.propStf.bixelWidth     = 5;
pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter       = matRad_getIsoCenter(cst,ct,0);

% dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 5; % [mm]

% Optimization settings
pln.propSeq.runSequencing = 1;
pln.propOpt.runDAO        = 0;

%%

stf = matRad_generateStf(ct,cst,pln);
%%

dij = matRad_calcDoseInfluence(ct,cst,stf,pln);
%%

cst_overlap = matRad_setOverlapPriorities(cst);

cst_resized = matRad_resizeCstToGrid(cst_overlap,dij.ctGrid.x,  dij.ctGrid.y,  dij.ctGrid.z,...
    dij.doseGrid.x,dij.doseGrid.y,dij.doseGrid.z);


%%
wInit = ones(9669,1);
%%
tic;
resultGUI = matRad_fluenceOptimization(dij,cst,pln,wInit);
toc
