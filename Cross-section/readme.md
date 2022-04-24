#member_util.py
a python function for generating a single structural member with various cross-sections. An example for the rectangular cross-sections is implemented therein.

#Assignments (Please divide work among the group members):
1. Add functions for generating steel cross-sections (W, HP, L, WT, 2L, HSS, Pipe) to the CrossSections class. See the links attached at the end of this file for the detailed information of cross-sections.

Example of functions:
```python
def W(...):
   ...

def Pipe(..., Inner=True):
   # When Inner==True, create mesh for the inner side of the cross-section
   # When Inner==False, create outside only.
```


2. We also want to create concrete cross-sections. Collect information that can be used to create similar functions for concrete columns with parameterization (what kind of shapes should be implemented to generate "common" types of bridges, discuss with pictures). We'll discuss them during the next meeting.

3. We also want to create steel member connections. Collect information that can be used to create similar functions for connections with parameterization. We'll discuss them during the next meeting.

Related resources:

[AISC Steep shape database](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.aisc.org%2Fglobalassets%2Faisc%2Fmanual%2Fv15.0-shapes-database%2Faisc-shapes-database-v15.0.xlsx&wdOrigin=BROWSELINK)

[AISC Steel design table](https://www.aisc.org/globalassets/aisc/manual/v15.1-companion/v15.1_vol-2_design-tables.pdf)

[Explanations of typical steel cross-sections](https://www.stainless-structurals.com/download/Brochure_Structurals-and-Custom-shapes-web.pdf)

[US FHWA, Bridge Inspector's Reference Manual (BIRM)](https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi12049.pdf)