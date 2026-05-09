# Sessin 1 - PDF Extraction Baseline

The thing here is understanding They are'nt really documents, they're rendering instructions. 
When we look at the pages that were extracted, the PDF  pages are different than the actual page number of the document itself. This is something that has to be noted when it comes to buildign embeddings later on. 

The 3 observations flagged from the 1st session on understanding PDFS 

1. Observation 1: PDF page numbers ≠ document page numbers: The script says "page 5." The PDF reader agrees — physically, this is the 5th sheet of paper. But the content says it's labeled iii (page 3 in roman numerals). Front matter pages (covers, TOC, foreword) typically use roman numerals, and the real document page 1 starts somewhere later — usually after the TOC ends.

2. Observation 2: TOC content is poison for RAG
For examples lines such as 1.6.5 Lighting ...................................................................................... 129. If this gets chunked and embedded, it has almost no semantic meaning. A user asks "what does it say about lighting?" and retrieval might surface this TOC entry instead of the actual content on page 129. The chunk contains the word "Lighting" — high keyword overlap, plausibly high embedding similarity — but it has zero useful information.

3. Observation 3: Whitespace pollution
There are leading blank space lines when we printed Page 5 of the document, That's not a print formatting artifact — it's literally in the extracted text. PDFs often have lots of vertical whitespace (margins, headers, line spacing) that turns into newlines and spaces in the extracted text. This adds noise to embeddings without contributing meaning.

These are few observations that can be used to fix the accurancy of the system. 

The TOC has dotted lines which does'nt have meaning. Hence this embedded can not be that useful. This is something that needs to be noted for further changes in the future. 




# Session 2: Chunking

The PDF that we have has 180 pages of text. Now we cant embed the whole document - this would produce a simply 1 fat vector. But we will have to embedd into smalled pieces called chunks, that captures the meaning. 

Here is the thing:
The chunks can not be too small like of 100chars as they might loose information. Or too big for e.g. 5000 characters which will be blurry average of the topics.  Like what we are trying to ask might be diluted in the real meaning of the long context. 
Hence, one coherent idea per chunk, like for regulatiuon document like this we can take 1 clause or 1 paragraph as a starting. 

Overlap: This is crucial, now if we are taking a 1 paragraph in 1 chunck it might have some more information about a topic in its subsequent pragraph. Hence we need to chunk them in a way that they overlap informaion. This way we will have information covered. 

This is what a Character based chunker is about. Cutting sentences mid-word, spilit tebles. 

Whle reading through chunks, here are some of the known issues that we found:
-  Frmont matter pollution  - Mainly onlu has the heading whihc is a noise
- Too much shite spae heavy chunks - jsut oene line of text, but there a full blank page.
- Page boundry content splits. TOC pollution with dotted lines, number etc. 