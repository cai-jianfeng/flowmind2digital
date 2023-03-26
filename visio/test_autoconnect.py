# In[1] start visio app
from visio import visio_app

app = visio_app()
visio = app.get_app(name='Visio.Application')
doc = app.add_doc(doc_name='Basic Diagram.vst')
page = app.add_page(doc=doc, number=1)
stn = app.get_template(template_name='BASIC_M.VSS')
stn2 = app.get_template(template_name='FAULT_M.VSSX')
master = app.choose_shape(shape_name='Rectangle', template=stn)
shp1 = app.add_shape(page=page, master=master, x=6, y=7)
app.resize_shape(shp=shp1, w='100pt', h='50pt')
text = 'shape1'
app.shape_text(shp=shp1, text=text)
shp2 = app.add_shape(page=page, master=master, x=6.02, y=6.1)
app.resize_shape(shp=shp2, w='100pt', h='50pt')
text = 'shape2'
app.shape_text(shp=shp2, text=text)

# In[2]
# zip([2.99, 4.93, 7.11, 9.02], [5, 5.1, 5.05, 4.95])
shpa = app.add_shape(page=page, master=master, x=2.99, y=4.93)
app.resize_shape(shp=shpa, w='100pt', h='50pt')
text = 'I am 2.99'
app.shape_text(shp=shpa, text=text)

shpb = app.add_shape(page=page, master=master, x=5.03, y=5.1)
app.resize_shape(shp=shpb, w='100pt', h='50pt')
text = 'I am 4.93'
app.shape_text(shp=shpb, text=text)

shpc = app.add_shape(page=page, master=master, x=7.11, y=5.05)
app.resize_shape(shp=shpc, w='100pt', h='50pt')
text = 'I am 7.11'
app.shape_text(shp=shpc, text=text)

shpd = app.add_shape(page=page, master=master, x=9.02, y=4.95)
app.resize_shape(shp=shpd, w='100pt', h='50pt')
text = 'I am 9.02'
app.shape_text(shp=shpd, text=text)

# In[3]
shp3 = app.add_shape(page=page, master=master, x=5.9, y=3)
app.resize_shape(shp=shp3, w='100pt', h='50pt')
text = 'shape3'
app.shape_text(shp=shp3, text=text)
shp4 = app.add_shape(page=page, master=master, x=9.1, y=2.9)
app.resize_shape(shp=shp4, w='100pt', h='50pt')
text = 'shape4'
app.shape_text(shp=shp4, text=text)
master2 = app.choose_shape(shape_name='Diamond', template=stn)
shp5 = app.add_shape(page=page, master=master2, x=2.9, y=3.1)
app.resize_shape(shp=shp5, w='100pt', h='50pt')
text = 'shape5'
app.shape_text(shp=shp5, text=text)
shp6 = app.add_shape(page=page, master=master, x=2.9, y=0.5)
app.resize_shape(shp=shp6, w='100pt', h='50pt')
text = 'shape6'
app.shape_text(shp=shp6, text=text)

shp7 = app.add_shape(page=page, master=master2, x=2.9, y=-1.5)
app.resize_shape(shp=shp7, w='100pt', h='50pt')
text = 'shape7'
app.shape_text(shp=shp7, text=text)

shp8 = app.add_shape(page=page, master=master, x=6.01, y=-1.51)
app.resize_shape(shp=shp8, w='100pt', h='50pt')
text = 'shape8'
app.shape_text(shp=shp8, text=text)

shp9 = app.add_shape(page=page, master=master, x=9.01, y=-1.49)
app.resize_shape(shp=shp9, w='100pt', h='50pt')
text = 'shape9'
app.shape_text(shp=shp9, text=text)

shp10 = app.add_shape(page=page, master=master, x=2.91, y=-2.5)
app.resize_shape(shp=shp10, w='100pt', h='50pt')
text = 'shape10'
app.shape_text(shp=shp10, text=text)

shp11 = app.add_shape(page=page, master=master2, x=9, y=-2.49)
app.resize_shape(shp=shp11, w='100pt', h='50pt')
text = 'shape11'
app.shape_text(shp=shp11, text=text)

shp12 = app.add_shape(page=page, master=master, x=12, y=-2.51)
app.resize_shape(shp=shp12, w='100pt', h='50pt')
text = 'shape12'
app.shape_text(shp=shp12, text=text)

shp13 = app.add_shape(page=page, master=master, x=2.9, y=-3.5)
app.resize_shape(shp=shp13, w='100pt', h='50pt')
text = 'shape13'
app.shape_text(shp=shp13, text=text)

shp14 = app.add_shape(page=page, master=master, x=2.91, y=-4.49)
app.resize_shape(shp=shp14, w='100pt', h='50pt')
text = 'shape14'
app.shape_text(shp=shp14, text=text)
# In[]
app.auto_connect(shp1=shp1, shp2=shp2, style='down', connect=None)
app.auto_connect(shp1=shp2, shp2=shpa, style='down', connect=None)
app.auto_connect(shp1=shp2, shp2=shpb, style='down', connect=None)
app.auto_connect(shp1=shp2, shp2=shpc, style='down', connect=None)
app.auto_connect(shp1=shp2, shp2=shpd, style='down', connect=None)
app.auto_connect(shp1=shpa, shp2=shp5, style='down', connect=None)
app.auto_connect(shp1=shpb, shp2=shp3, style='down', connect=None)
app.auto_connect(shp1=shpc, shp2=shp3, style='down', connect=None)
app.auto_connect(shp1=shpd, shp2=shp3, style='down', connect=None)
app.auto_connect(shp1=shp5, shp2=shp3, style='right', connect=None)
app.auto_connect(shp1=shp3, shp2=shp4, style='right', connect=None)
app.auto_connect(shp1=shp5, shp2=shp6, style='down', connect=None)
app.auto_connect(shp1=shp6, shp2=shp7, style='down', connect=None)
app.auto_connect(shp1=shp7, shp2=shp8, style='right', connect=None)
app.auto_connect(shp1=shp8, shp2=shp9, style='right', connect=None)
app.auto_connect(shp1=shp7, shp2=shp10, style='down', connect=None)
app.auto_connect(shp1=shp9, shp2=shp11, style='down', connect=None)
app.auto_connect(shp1=shp11, shp2=shp10, style='left', connect=None)
app.auto_connect(shp1=shp11, shp2=shp12, style='right', connect=None)
app.auto_connect(shp1=shp10, shp2=shp13, style='down', connect=None)
app.auto_connect(shp1=shp13, shp2=shp14, style='down', connect=None)
