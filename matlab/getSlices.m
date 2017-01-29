function y = getSlices(dir, isNii, isADNI)
    outputFolder = 'O:\Output\';
    sliceNo = 70;
    
    if ~isdir(outputFolder)
        errorMessage = sprintf('Error: Please create output folder:\n%s', outputFolder);
        uiwait(warndlg(errorMessage));
        return;
    end
    
    docNode = com.mathworks.xml.XMLUtils.createDocument('metadata');
    
    if isNii == true
        niiFiles = rdir([dir, '\**\*.nii']);
        toc = docNode.getDocumentElement;

        for i = 1:numel(niiFiles);
            fprintf('Reading file no: %d out of %d\n', i, numel(niiFiles));
            [pathstr,name,ext] = fileparts(niiFiles(i).name);
            s = name;
            if isADNI == true
                xmlName = strcat(s(1: strfind(s, '_MR')), s(strfind(s, '_MR') + 4:strfind(s, '_Br')), s(strfind(s, '_Br') + 22:numel(s)));
            else
                xmlName = s;
            end
            disp(strcat(xmlName, '.xml'));
            xmlName = sprintf('\\**\\*%s.xml', xmlName);
            xmlFile = rdir([dir, xmlName]);

            if (numel(xmlFile) > 0)
                image = docNode.createElement('image');
                image.setAttribute('name',s);

                xDoc = xmlread(xmlFile(1).name);
                
                allAssessments = xDoc.getElementsByTagName('assessmentScore');

                for k = 0:allAssessments.getLength-1
                    thisElement = allAssessments.item(k);
                    if (strcmp(char(thisElement.getAttribute('attribute')), 'MMSCORE'))
                       MMSEres = char(thisElement.getFirstChild.getData);
                    end
                end

                if and(exist('MMSEres', 'var'), ~isempty(MMSEres))
                    mmse = docNode.createElement('MMSE');
                    mmse.appendChild(docNode.createTextNode(MMSEres));
                    image.appendChild(mmse);
                else
                    mmse = docNode.createElement('MMSE');
                    mmse.appendChild(docNode.createTextNode('30'));
                    image.appendChild(mmse);
                end

                subSexEl = xDoc.getElementsByTagName('subjectSex');
                disp(subSexEl.item(0));
                sex = char(subSexEl.item(0).getFirstChild.getData);
                if ~isempty(subSexEl)
                    sexEl = docNode.createElement('subjectSex');
                    sexEl.appendChild(docNode.createTextNode(sex))
                    image.appendChild(sexEl);
                end

                subAgeEl = xDoc.getElementsByTagName('subjectAge');
                age = char(subAgeEl.item(0).getFirstChild.getData);
                if ~isempty(subAgeEl)
                    ageEl = docNode.createElement('subjectAge');
                    ageEl.appendChild(docNode.createTextNode(age));
                    image.appendChild(ageEl);
                end 
                toc.appendChild(image);
                
                protocols = xDoc.getElementsByTagName('protocol');
                for iter = 0:protocols.getLength-1
                    protocol = protocols.item(iter);
                    if (strcmp(char(protocol.getAttribute('term')), 'Matrix X'))
                        xSize = str2double(char(protocol.getFirstChild.getData));
                    end
                end
                
                if xSize == 256
                    continue;
                end
            else
                image = docNode.createElement('image');
                image.setAttribute('name',s);
                mmse = docNode.createElement('MMSE');
                mmse.appendChild(docNode.createTextNode('30'));
                image.appendChild(mmse); 
                toc.appendChild(image);
            end
            
            %sliceNiiFile(sliceNo, niiFiles(i), outputFolder);
        end
    end
    
    if and(isNii == false, isADNI == false)
        name = sprintf('\\**\\*%d.dcm', sliceNo);
        dcmFiles = rdir([dir, name]);
        for j = 1:numel(dcmFiles);
            fprintf('Reading file no: %d out of %d\n', j, numel(dcmFiles));
            disp(dcmFiles(j).name);
            image = docNode.createElement('image');
            mmse = docNode.createElement('MMSE');
            mmse.appendChild(docNode.createTextNode('30'));
            image.appendChild(mmse);
            
            image.setAttribute('name',dcmFiles(j).name);
            toc.appendChild(image);
            
            sliceDcmFile(dcmFiles(j), outputFolder);
        end
    end

    filename = sprintf('metadata_%s.xml', datestr(now, 'HH_MM_SS_FFF'));
    
    fclose(fopen(filename, 'w'));
    
    xmlwrite(filename, docNode);
    type(filename);
    
    y = 'OK';
end