function out = MF_NCR_b(seed,rawData,file_folder,varargin)
rng(seed);
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',10,@(x) isnumeric(x));
params.addParameter('lr',10,@(x) isnumeric(x));
params.addParameter('regU',0.01,@(x) isnumeric(x));
params.addParameter('regV',0.01,@(x) isnumeric(x));
params.addParameter('regB',0.01,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('maxIter',200,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run biasedMF and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
par.m = max(rawData(:,1));
par.n = max(max(rawData(:,2)));
temp = arrayfun(@(x) rawData(x,1),(1:length(rawData))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    [U,V,Bu,theta] = feval(methodSolver,rawData,trainIdx,testIdx,par);
    out(i,:) = MFRankEval_NCR_b(rawData,testIdx,U,V,Bu,theta,par);
    filename_1 = sprintf('MF_NCR_b_fold_%i.mat',i);
    filename = fullfile(file_folder,filename_1);
    save (filename,'testIdx','U','V','theta','Bu','par','-mat');
    fprintf('MF_b NCR b %d/%d fold completed\n',i,cvObj.NumTestSets);
end
fprintf('final result:AUC = %f, NDCG = %f, RMSE = %f, MAE = %f, MRR = %f \n',mean(out));
result = sprintf('final result:AUC = %f, NDCG = %f, RMSE = %f, MAE = %f, MRR = %f \n',mean(out));
filename = fullfile(file_folder,'MF_NCR_b_final_result.mat');
save (filename,'out','result','-mat');
end

function [U,V,Bu,theta] = graded_solver(rawData,trainIdx,testIdx,par)
trainData = rawData(trainIdx,:);
trainData = trainData(randperm(size(trainData,1)),:);
fprintf('generate data completed\n');
theta = 0.1;
U = normrnd(0,0.1,par.m,par.F);
V = normrnd(0,0.1,par.n,par.F);
Bu = normrnd(0,0.01,par.m,par.F);
incU = zeros(par.m,par.F);
incV = zeros(par.n,par.F);
incbu = zeros(par.m,par.F);
incTheta = 0;
lastLoss = 0;
for i = 1:par.maxIter
    loss = 0;
    u = U(trainData(:,1),:);
    v = V(trainData(:,2),:);
    bu = Bu(trainData(:,1),:);
    x1 = zeros(length(trainIdx),par.F);
    x2 = zeros(length(trainIdx),par.F);
    ixV = zeros(length(trainIdx),par.F);
    ixTheta = zeros(length(trainIdx),1);
    sumU = sum(exp(u),2);
    for m = 1:par.F
        idx_diff_k = setdiff([1:par.F],m);
        x1(:,m) = exp(u(:,m))./sumU;
        x2(:,m) = (exp(theta).*v(:,m))+sum(v(:,idx_diff_k)-bu(:,idx_diff_k),2);
        ixV(:,m) = (exp(u(:,m)).*exp(theta)+(sumU-exp(u(:,m))))./sumU;
        ixTheta = ixTheta + (x1(:,m).*exp(theta).*v(:,m));       
    end   
    pred = sum(x1.*x2,2);
    ixU = x2.*(x1-x1.^2);
    ixbu = -(1-x1);
    error = pred-trainData(:,3);
    loss = loss+sum(error.^2);
    ixU = error.*ixU+par.regU.*u;
    ixV = error.*ixV+par.regV.*v;
    ixbu = error.*ixbu;
    ixTheta = error.*ixTheta;
    gU = zeros(par.m,par.F);
    gV = zeros(par.n,par.F);
    gbu = zeros(par.m,par.F);
    gTheta = sum(ixTheta);
    for z = 1:length(trainIdx)
        gU(trainData(z,1),:) = gU(trainData(z,1),:)+ixU(z,:);
        gV(trainData(z,2),:) = gV(trainData(z,2),:)+ixV(z,:);
        gbu(trainData(z,1),:) = gbu(trainData(z,1),:)+ixbu(z,:);
    end
    incU = par.momentum*incU+par.lr*gU/length(trainIdx);
    incV = par.momentum*incV+par.lr*gV/length(trainIdx);
    incbu = par.momentum*incbu+par.lr*gbu/length(trainIdx);
    incTheta = par.momentum*incTheta+par.lr*gTheta/length(trainIdx);
    U = U - incU;
    V = V - incV;
    Bu = Bu - incbu;

    theta = theta - incTheta;
    if theta<0
        theta=0.1;
    end
    loss = loss+par.regU*sum(sum(U.^2))+par.regV*sum(sum(V.^2));%loss = loss+par.regU*sum(sum(U(trainData(:,1),:).^2))+par.regV*sum(sum(V(trainData(:,2),:).^2));
    deltaLoss = lastLoss-0.5*loss;
%     if abs(deltaLoss)<1e-5
%         break;
%     end

    out = MFRankEval_NCR_b(rawData,testIdx,U,V,Bu,theta,par);
%     lastLoss = 0.5*loss;
    if par.adaptive && i > 2
        if lastLoss > 0.5*loss
            par.lr = 1.05*par.lr;
        else
            par.lr = 0.7*par.lr;
        end
    end
    lastLoss = 0.5*loss;
    if mod(i,10)==0
        fprintf('MF_NCR_bU iter [%d/%d] completed, loss = %f, delta_loss: %f, lr: %f theta=%f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr,theta);
        fprintf('AUC = %f, NDCG = %f, RMSE = %f, MAE = %f, MRR = %f \n',out);
    end
end
end