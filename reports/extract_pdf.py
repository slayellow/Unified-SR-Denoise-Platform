import fitz

doc = fitz.open('/home/jshong/Code/unfied-sr-denoise-platform/reports/260414_Denoise_Comparison.pdf')
print("Pages:", len(doc))
for i, page in enumerate(doc):
    page.get_pixmap().save(f'/home/jshong/Code/unfied-sr-denoise-platform/reports/page_{i}.png')
