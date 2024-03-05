function doneOr = checkStatus_local(jobID)
%--------------------------------------------------------------------
%This routine is to check if the submitted job is done or not
%One needs to do a little edit based on your own case.
%1   : whichCluster (0: no-job-script, 1: local submission, 2: remote submission)
%--------------------------------------------------------------------

%Step1: the command to check job by ID. 
    [a,b] = unix(['qstat ' jobID '']) ;

%Step2: to find the keywords from screen message to determine if the job is done
%Below is just a sample:

% Job id              Name             Username        Time Use S Queue
% ------------------- ---------------- --------------- -------- - ---------------
% 3441658             c-ch2choch2_ch2c jcloison        00:00:04 R ism

%If the job is still running, it will show as above.

%If there is no key words like 'R/Q Cfn_gen04', it indicates the job is done.
    if isempty(findstr(b,'R ircer')) & isempty(findstr(b,'Q ircer')) & isempty(findstr(b,'C ircer')) 
       doneOr = 1
       [nothing, nothing] = unix('rm USPEX*');    % to remove the log file
% TO DO : DECIDE WHAT TO DO WHEN JOB IS IN "C" STATUS
    else
       doneOr = 0;
    end
