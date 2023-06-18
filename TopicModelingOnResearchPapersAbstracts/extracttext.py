import PyPDF2
import sys
import os

class MyExtractor:
    replaceTerms = {}
    abstracts = []
    abstractStoppers = ['index terms',
                        'the authors',
                        'ccs concepts',
                        'introduction',
                        'keywords',
                        'list of papers']
    abstractStarters = ['abstract',
                        'a b s t r a c t',
                        'summary']

    def convertToTextPyPDF2(self, pdfFile, writeToFile):
        import re
        
        abstractEnd = False
        abstractFound = False
        with open(pdfFile, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            num_pages = pdf_reader.numPages

            abstract = ""
            words = ""

            for i in range(num_pages):
                page = pdf_reader.getPage(i)
                text = page.extract_text()

                last = ""
                for w in text.split():
                    w = w.replace('.','').replace(',','').replace(' ', '')
                    w = re.sub('[^a-zA-Z]', '', w).lower()
                    w = w.strip()
                    if len(w) == 0:
                        continue

                    if not abstractFound:
                        for starter in self.abstractStarters:
                            if w.startswith(starter) or words.endswith(starter):
                                abstractFound = True
    #                            print('Found abstract')
                                break
                        if not abstractFound:
                            words = words + w
                    else:
                        abstract = abstract + ' ' + w
    #                    if w == 'work':
    #                        print('found term')
                        for endword in self.replaceTerms.keys():
                            if abstract.endswith(endword):
                                abstract = abstract.replace(endword, self.replaceTerms[endword])
                                break
                        for stopper in self.abstractStoppers:
                            if abstract.endswith(stopper):
                                abstract = abstract.replace(stopper, '')
                                abstractEnd = True
    #                            print('Found abstract end')
                                break
                    
                    if abstractEnd:
                        break
                
                if abstractEnd:
                    break
        
        if abstractEnd and writeToFile:
            file = open(pdfFile.replace('.pdf','.txt'), 'w')
            file.write(abstract)
            file.close

        return abstract

    def extractAbstractsUsingPyPDF2(self, writeToFile, delTextFile):
        self.replaceTerms['t erms'] = 'terms'
        self.replaceTerms['t ermsmicroservice'] = 'terms'
        self.replaceTerms['i i ntroduction'] = 'introduction'
        self.replaceTerms['microservicebased'] = 'microservice based'
        self.replaceTerms['microservicesbased'] = 'microservices based'
        self.replaceTerms['automationheavy'] = 'automation heavy'
        self.replaceTerms['moni toring'] = 'monitoring'
        self.replaceTerms['restapi'] = 'rest api'
        self.replaceTerms['afunctional'] = 'a functional'
        self.replaceTerms['desig ned'] = 'designed'
        self.replaceTerms['p ractically'] = 'practically'
        self.replaceTerms['differ ent'] = 'different'
        self.replaceTerms['de vices'] = 'devices'
        self.replaceTerms['u sers'] = 'users'
        self.replaceTerms['th is'] = 'this'
        self.replaceTerms['crossservice'] = 'cross service'
        self.replaceTerms['networkendpointbased'] = 'network endpoint based'
        self.replaceTerms['network independent'] = 'network independent'
        self.replaceTerms['perime terization'] = 'perimeterization'
        self.replaceTerms['per packet'] = 'per packet'
        self.replaceTerms['ap proach'] = 'approach'
        self.replaceTerms['proofofconcept'] = 'proof of concept'
        self.replaceTerms['tradi tional'] = 'traditional'
        self.replaceTerms['severalfactorsexplainthisdicultyincludingaknowledgegapamongmicroservicesprac titionersonproperlysecuringamicroservicessystem'] = 'several factors explain this difficulty including a knowledge gap among microservices practitioners on properly securing a microservices system'
        self.replaceTerms['topartiallybridgethisgapweconductedan empiricalstudy'] = 'to partially bridge this gap we conducted an empirical study'
        self.replaceTerms['werstmanuallyanalyzedmicroservicessecuritypointsincludingissues'] = 'we manually analyzed microservices security points including issues'
        self.replaceTerms['pointisreferredtoasagithubissueastackoverowpostadocumentorawikipagethatentails'] = 'point is referred to as a github issue a stackoverflow post a document or a wiki page that entails'
        self.replaceTerms['ormoremicroservicessecurityparagraphs'] = 'or more microservices security paragraphs'
        self.replaceTerms['ouranalysisledtoacatalogofmicroservicessecurity'] = 'our analysis led to a catalog of microservices security'
        self.replaceTerms['wethenranasurveywithmicroservicespractitionerstoevaluatetheusefulnessofthese'] = 'we then ran a survey with microservices practitioners to evaluate the usefulness of these'
        self.replaceTerms['ndings'] = 'findings'
        self.replaceTerms['pop ular'] = 'popular'
        self.replaceTerms['microser vice'] = 'microservice'
        self.replaceTerms['mi croservice'] = 'microservice'
        self.replaceTerms['prac titioners'] = 'practitioners'
        self.replaceTerms['se curity'] = 'security'
        self.replaceTerms['develop ment'] = 'develop ment'
        self.replaceTerms['o ur'] = 'our'
        self.replaceTerms['de ployment'] = 'deployment'
        self.replaceTerms['data set'] = 'dataset'
        self.replaceTerms['trafc'] = 'traffic'
        self.replaceTerms['identied'] = 'identified'
        self.replaceTerms['sufciently'] = 'sufficiently'
        self.replaceTerms['difcult'] = 'difficult'
        self.replaceTerms['defenceindepth'] = 'defence in depth'
        self.replaceTerms['continu ously'] = 'continuously'
        self.replaceTerms['criti cal'] = 'critical'
        self.replaceTerms['costsensitive'] = 'cost sensitive'
        self.replaceTerms['speci c'] = 'specific'
        self.replaceTerms['ser vices'] = 'services'
        self.replaceTerms['al ternatives'] = 'alternatives'
        self.replaceTerms['availabil ity'] = 'availabil ity'
        self.replaceTerms['tac tics'] = 'tactics'
        self.replaceTerms['bibli ographic'] = 'bibliographic'
        self.replaceTerms['relevan t'] = 'relevant'
        self.replaceTerms['machinelearning'] = 'machinelearning'
        self.replaceTerms['inbetween'] = 'in between'
        self.replaceTerms['efcient'] = 'efficient'
        self.replaceTerms['re lationship'] = 'relationship'
        self.replaceTerms['toaddress'] = 'to address'
        self.replaceTerms['subgridoriented'] = 'sub grid oriented'
        self.replaceTerms['welldesigned'] = 'well designed'
        self.replaceTerms['spatialtemporal'] = 'spatial temporal'
        self.replaceTerms['acmodel'] = 'ac model'
        self.replaceTerms['spatialtemporalrelationship'] = 'spatial temporal relationship'
        self.replaceTerms['amicroservice'] = 'a microservice'
        self.replaceTerms['proposedfor'] = 'proposed for'
        self.replaceTerms['fromsubgrids'] = 'from subgrids'
        self.replaceTerms['datasetsare'] = 'datasets are'
        self.replaceTerms['methodoutperforms'] = 'method out performs'
        self.replaceTerms['stateoftheart'] = 'state of the art'
        self.replaceTerms['inthese'] = 'in these'
        self.replaceTerms['char acteristics'] = 'characteristics'
        self.replaceTerms['softwarearchitect'] = 'software architect'
        self.replaceTerms['thedesign'] = 'the design'
        self.replaceTerms['errorprone'] = 'error prone'
        self.replaceTerms['scenariosthis'] = 'scenarios this'
        self.replaceTerms['approachfor'] = 'approach for'
        self.replaceTerms['automaticallyidentify'] = 'automatically identify'
        self.replaceTerms['beextended'] = 'be extended'
        self.replaceTerms['addressedin'] = 'addressed in'
        self.replaceTerms['reliableand'] = 'reliable and'
        self.replaceTerms['proposea'] = 'propose a'
        self.replaceTerms['systemsthat'] = 'systems that'
        self.replaceTerms['vulnerableagainst']= 'vulnerable against'
        self.replaceTerms['operatingsystems'] = 'operating systems'
        self.replaceTerms['evaluationour'] = 'evaluation our'
        self.replaceTerms['v ulnerabilities'] = 'vulnerabilities'
        self.replaceTerms['cyberrisk'] = 'cyber risk'
        self.replaceTerms['microservicessuch'] = 'microservices such'
        self.replaceTerms['congurations'] = 'configurations'
        self.replaceTerms['mostvulnerable'] = 'most vulnerable'
        self.replaceTerms['popularoperating'] = 'popular operating'
        self.replaceTerms['t ermsmicroservices'] = 'terms microservices'
        self.replaceTerms['appli cations'] = 'applications'
        self.replaceTerms['couplingunfortunately'] = 'coupling unfortunately'
        self.replaceTerms['in terconnected'] = 'interconnected'
        self.replaceTerms['softwarethatwebuildshouldbesecureresilientandreliablebothagainstaccidents'] = 'software that we build should be secure resilient and reliable both against accidents'
        self.replaceTerms['themicroservicearchitectureorconcisely'] = 'the microservice architecture or concisely'
        self.replaceTerms['isarecenttrendinsoft ware'] = 'is a recent trend in software'
        self.replaceTerms['introduc ing'] = 'introducing'
        self.replaceTerms['wedesignataxonomyofmicroservicesecuritygivinganoverviewoftheexisting'] = 'we design a taxonomy of microservice security giving an overview of the existing'
        self.replaceTerms['mi croservicesecuritytrendsinindustry'] = 'microservice security trends in industry'
        self.replaceTerms['furthermorewepresentanopensourcepro totype'] = 'further more we present an opensource prototype'
        self.replaceTerms['wetakethedefenseindepthprincipleevenfurtherbyfocusingourattentionon'] = 'we take the defense in depth principle even further by focusing our attention on'
        self.replaceTerms['selfprotection'] = 'self protection'
        self.replaceTerms['serviceoriented sys tem'] = 'service oriented system'
        self.replaceTerms['securityasa'] = 'security as a'
        self.replaceTerms['infras tructure'] = 'infrastructure'
        self.replaceTerms['exible'] = 'flexible'
        self.replaceTerms['vulnera ble'] = 'vulnerable'
        self.replaceTerms['rene'] = 'refine'
        self.replaceTerms['renement'] = 'refinement'
        self.replaceTerms['rening'] = 'refining'
        self.replaceTerms['elds'] = 'fields'
        self.replaceTerms['hostcontainer'] = 'host container'
        self.replaceTerms['softwarebased'] = 'software based'
        self.replaceTerms['hardwarebased'] = 'hardware based'
        self.replaceTerms['missioncritical'] = 'mission critical'
        self.replaceTerms['differ ent'] = 'different'
        self.replaceTerms['wellde nedcorpus'] = 'well defined corpus'
        self.replaceTerms['speci c'] = 'specific'
        self.replaceTerms['highperformance'] = 'high performance'
        self.replaceTerms['simpletounderstand'] = 'simple to understand'
        self.replaceTerms['networkbased'] = 'network based'
        self.replaceTerms['lightw eight'] = 'light weight'
        self.replaceTerms['micros ervice'] = 'microservice'
        self.replaceTerms['vari ous'] = 'various'
        self.replaceTerms['developmentmanagement'] = 'development management'
        self.replaceTerms['conti nuous'] = 'continuous'
        self.replaceTerms['oftheart'] = 'of the art'
        self.replaceTerms['keyw ords'] = 'keywords'
        self.replaceTerms['su pport'] = 'support'
        self.replaceTerms['availabilityresiliency'] = 'availability resiliency'
        self.replaceTerms['peerreviewed'] = 'peer reviewed'

        if len(sys.argv) == 2:
            self.abstracts.append(self.convertToTextPyPDF2(sys.argv[1], writeToFile))
        else:
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            for f in files:
                if f.endswith(".pdf"):
                    if delTextFile:
                        txtFile = f.replace('.pdf', '.txt')
                        if os.path.exists(txtFile):
                            os.remove(txtFile)
                            print(txtFile, ' deleted!')

                    print("Extracting abstract from " + f)
                    self.abstracts.append(self.convertToTextPyPDF2(f, writeToFile))
        return self.abstracts
