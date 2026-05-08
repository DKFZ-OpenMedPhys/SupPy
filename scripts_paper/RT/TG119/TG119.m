%% Example: Proton Treatment Plan with subsequent Isocenter shift
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2017 the matRad development team.
%
% This file is part of the matRad project. It is     subject to the license
% terms in the LICENSE file found in the top-level directory of this
% distribution and at https://github.com/e0404/matRad/LICENSE.md. No part
% of the matRad project, including this file, may be copied, modified,
% propagated, or distributed except according to the terms contained in the
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% In this example we will show
% (i) how to load patient data into matRad
% (ii) how to setup a proton dose calculation
% (iii) how to inversely optimize the pencil beam intensities directly from command window in MATLAB.
% (iv) how to simulate a lateral patient displacement by shifting the iso-center
% (v) how to recalculated the dose considering the shifted geometry and the previously optimized pencil beam intensities
% (vi) how to compare the two results

%% set matRad runtime configuration
matRad_rc; %If this throws an error, run it from the parent directory first to set the paths
%% Patient Data Import
% Let's begin with a clear Matlab environment and import the prostate
% patient into your workspace

load('TG119.mat');


%%

cst{1,6}{1} = DoseConstraints.matRad_MinMaxDVH(10,0,5);
%cst{2,6} = [];
%cst{
cst{2,6}{1} = DoseConstraints.matRad_MinMaxDose(58,68);
cst{2,6}{2} = DoseConstraints.matRad_MinMaxDVH(60,95,100);
cst{2,6}{3} = DoseConstraints.matRad_MinMaxDVH(65,0,5);

cst{3,6} = {};
cst{3,6}{1} = DoseObjectives.matRad_MeanDose(100,0);

pln.radiationMode   = 'photons';
pln.machine         = 'Generic';
pln.numOfFractions  = 30;

pln.bioModel = 'none';
pln.multScen = 'nomScen';


pln.propStf.gantryAngles   = [0:40:359];
pln.propStf.couchAngles    = zeros(1,numel(pln.propStf.gantryAngles));
pln.propStf.bixelWidth     = 5;
pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter       = matRad_getIsoCenter(cst,ct,0);

% dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 2.5; % [mm]

% Optimization settings
pln.propSeq.runSequencing = 1;
pln.propOpt.runDAO        = 0;

%% Generate Beam Geometry STF
stf = matRad_generateStf(ct,cst,pln);
dij = matRad_calcDoseInfluence(ct,cst,stf,pln);

cst_overlap = matRad_setOverlapPriorities(cst);

cst_resized = matRad_resizeCstToGrid(cst_overlap,dij.ctGrid.x,  dij.ctGrid.y,  dij.ctGrid.z,...
    dij.doseGrid.x,dij.doseGrid.y,dij.doseGrid.z);
