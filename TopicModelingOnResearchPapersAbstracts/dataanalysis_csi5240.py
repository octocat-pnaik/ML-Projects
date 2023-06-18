from lda import MyLDA
from extracttext import MyExtractor
import time

if __name__ == "__main__":
    start = time.time()

    analyze = True
    writeTextFile = False
    delTextFile = True
    numTopics = 20
    
    print('Start data analysis, extracting abstracts from PDFs ...')
    extractor = MyExtractor()
    abstracts = extractor.extractAbstractsUsingPyPDF2(writeTextFile, delTextFile)
    print('Number of abstracts: ', len(abstracts))

    if analyze:
        print('Starting LDA analysis ...')
        ldaAnalysis = MyLDA()
        ldaAnalysis.do_LDA_analysis_texts(abstracts, numTopics, 'abstracts', None, None, None) 
        print('Done')

    print('Execution time: ', time.time()-start)
