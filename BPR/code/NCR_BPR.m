function [U,V,Theta] = NCR_BPR(seed,session,file_folder,varargin)
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('lr',5,@(x) isnumeric(x));
% params.addParameter('lrTheta',50,@(x) isnumeric(x));
params.addParameter('regU',0.3,@(x) isnumeric(x));%0.01
params.addParameter('regV',0.3,@(x) isnumeric(x));
params.addParameter('regTheta',0,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
% params.addParameter('alpha',1,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('maxIter',100,@(x) isnumeric(x));%100
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));
params.addParameter('earlyStop',true,@(x) islogical(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
par.m = session{end}.allUser;
par.n = session{end}.allItem;
session(end) = [];
%% Run BTR and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
%     tic
    [U,V,Theta] = feval(methodSolver,session,trainIdx,testIdx,par);
%     toc
    filename_1 = sprintf('BPR_NCR_fold_%i.mat',i);
    filename = fullfile(file_folder,filename_1);
    save (filename,'testIdx','U','V','Theta','par','-mat');
    out(i,:) = NCR_BPRRankEval(session,testIdx,U,V,Theta,par);
    fprintf('BPR:fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end
fprintf('BPR Final Results: auc = %f, ndcg = %f, MRR = %f, map = %f, pre = %f\n',mean(out));
result = sprintf('BPR Final Results: auc = %f, ndcg = %f, MRR = %f, map = %f, pre = %f\n',mean(out));
filename = fullfile(file_folder,'final_result.mat');
save (filename,'out','result','-mat');
end

function [U,V,Theta] = graded_solver(session,trainIdx,testIdx,par)
D = cell(length(trainIdx),1);
for i = 1:length(trainIdx)
    sample = session{trainIdx(i)};
    buyItem = sample.buy(1,:);
    noBuyItem = sample.noBuy(1,:);
    comparePair = combvec(buyItem,noBuyItem);
    D{i} = [repmat(sample.user,[size(comparePair,2),1]),comparePair'];
end
D = cell2mat(D);
batchIdx = discretize(1:size(D,1),par.batchNum);
[~,p] = numunique(batchIdx);
fprintf('BPR_NCR generate data completed\n');

U = rand(par.m,par.F);
V = rand(par.n,par.F);
Theta = 1;
oldU = U;oldV = V;
incU = zeros(par.m,par.F);
incW = zeros(par.n,par.F);
incV = zeros(par.n,par.F);
incTheta = 0;

bestAUC = 0;
loseNum = 0;
lastLoss = 0;

for i = 1:par.maxIter
    loss = 0;
    tic
    for j = 1:par.batchNum
        u = U(D(p{j},1),:);
        w = V(D(p{j},2),:);
        v = V(D(p{j},3),:);
        x = zeros(length(D(p{j})),1);
        ixU = zeros(length(D(p{j})),par.F);
        ixW = zeros(length(D(p{j})),par.F);
        ixV = zeros(length(D(p{j})),par.F);
        ixTheta = zeros(length(D(p{j})),1);
        for m = 1:par.F
            x1 = exp(u(:,m))./sum(exp(u),2);
            x2 = exp(Theta).*(w(:,m)-v(:,m))+sum(w,2)-w(:,m)-sum(v,2)+v(:,m);
            x = x + x1.*x2;
            ixU(:,m) = x2.*(x1-x1.^2);
            ixW(:,m)=(exp(u(:,m)).*exp(Theta)+sum(exp(u),2)-exp(u(:,m)))./sum(exp(u),2);
            ixV(:,m)= -ixW(:,m);
            ixTheta = ixTheta + x1.*(exp(Theta).*(w(:,m)-v(:,m)));
        
        end
        ix1 = -logsig(-x);
        ixU = ix1.*ixU+par.regU*U(D(p{j},1),:);
        ixW = ix1.*ixW+par.regV*V(D(p{j},2),:);
        ixV = ix1.*ixV+par.regU*V(D(p{j},3),:);
        ixTheta = ix1.*ixTheta+par.regTheta*Theta;
        
        loss = loss+sum(-log(logsig(x)));
        
        gU = zeros(par.m,par.F);
        gW = zeros(par.n,par.F);
        gV = zeros(par.n,par.F);
        gTheta = sum(ixTheta);
        for z = 1:length(p{j})
            gU(D(p{j}(z),1),:) = gU(D(p{j}(z),1),:)+ixU(z,:);
            gW(D(p{j}(z),2),:) = gW(D(p{j}(z),2),:)+ixW(z,:);
            gV(D(p{j}(z),3),:) = gV(D(p{j}(z),3),:)+ixV(z,:);
        end
        
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incW = par.momentum*incW+par.lr*gW/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incTheta = par.momentum*incTheta+par.lr*gTheta/length(p{j});

        U = U - incU;
        V = V - incW;
        V = V - incV;
        Theta = Theta - incTheta;

%         U = U - par.alpha.*incU;
%         V = V - par.alpha.*incW;
%         V = V - par.alpha.*incV;
%         Theta = Theta - par.alpha*incTheta;
% 
        if Theta<0
            Theta=0.1;
        end
        
        
        loss = loss+par.regU*sum(sum(U(D(p{j},1),:).^2))+par.regV*sum(sum(V(D(p{j},2),:).^2))+...
            par.regV*sum(sum(V(D(p{j},3),:).^2));
%         if j==1
%             fprintf('BPR_NCR:');
%         end
%         fprintf(' %d',Theta);
        
    end

    deltaLoss = lastLoss-0.5*loss;
    if abs(deltaLoss)<1e-5
        fprintf('========stop1===========\n');
        break;
    end
    
    cU=(oldU-U).^2;cV=(oldV-V).^2;
    if abs(sqrt(sum(cU(:))))<1e-4 || abs(sqrt(sum(cV(:))))<1e-4
        fprintf('=========stop2==========\n');
    	break;
    end
    oldU = U;oldV = V;
    
    out = NCR_BPRRankEval(session,testIdx,U,V,Theta,par);
%     if par.earlyStop
%         if out(1) <= bestAUC || isinf(exp(Theta)) || isnan(exp(Theta))
%             loseNum = loseNum+1;
% %             U = bestU;
% %             V = bestV;
% %             Theta = bestTheta;
%             
% %             incU = best_incU;
% %             incW = best_incW;
% %             incV = best_incV;
% %             incTheta = best_incTheta;
% %             
% %             par.alpha = 0.5*par.alpha;
%             if loseNum >= 10
%             	U = bestU;
%                 V = bestV;
%                 Theta = bestTheta;
%                 break;
%             end
%         else
%             bestAUC = out(1);
%             bestU = U;
%             bestV = V;
%             bestTheta = Theta;
% %             best_incU = incU;
% %             best_incW = incW;
% %             best_incV = incV;
% %             best_incTheta = incTheta;
%             loseNum = 0;
% %             if par.alpha<1
% %                 par.alpha = 1.2*par.alpha;
% %             end
%         end
%     end
    if par.adaptive && i > 2
        if lastLoss > 0.5*loss
            par.lr = 1.05*par.lr;
        else
            par.lr = 0.5*par.lr;
        end
        lastLoss = 0.5*loss;
    else
        lastLoss = 0.5*loss;
    end
%     if mod(i,1)==0
        fprintf('BPR_NCR:iter [%d/%d] completed, loss: %f, delta_loss = %f, lr: %f ,theta:%f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr,Theta);
        fprintf('BPR_NCR Results: auc = %f, ndcg = %f, MRR = %f, map = %f, pre = %f\n',out);
%     end
    toc
end
end