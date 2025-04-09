# download GSE217460 data
echo "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5FS01%2DS04%2Ecsv%2Egz
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%2Eh5ad%2Egz
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%5Fraw%2Eh5ad%2Egz
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fraw%2Eh5ad%2Egz
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fsubsample%2Eh5ad%2Egz
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fsubsample%5Fraw%2Eh5ad%2Egz" | xargs -n 2 -P 6 wget