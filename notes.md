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
