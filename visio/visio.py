# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: visio.py
@Date: 2022/11/29 23:26
@Author: caijianfeng
"""
import win32com.client as win32
from win32com.client import constants


class visio_app:
    def __init__(self):
        self.app = None

    def get_app(self, name):
        """
        start app(visio) using code
        :param name: visio name('Visio.Application')
        :return: app(type: IVApplication)
        """
        if not self.app or name:
            appVisio = win32.gencache.EnsureDispatch(name)
            self.app = appVisio

        return self.app

    def set_app_visible(self, visible, app=None):
        """
        set visio app is visible or not
        :param app: visio app(type: IVApplication)
        :param visible: (type:int) -> 1: visible; 0: invisible
        :return: None
        """
        if not app and self.app is not None:
            self.app.Visible = visible
        else:
            app.Visible = visible

    def get_template(self, template_name, app=None):
        """
        open visio template(built-in)
        :param template_name: template_name(type: str)
        :param app: visio app(type: IVApplication)
        :return: template(type: IVDocument)
        """
        if not app and self.app is not None:
            stn = self.app.Documents.Open(template_name)  # open template
        else:
            stn = app.Documents.Open(template_name)

        return stn

    def add_doc(self, doc_name, app=None):
        """
        create/open visio file/document you want(.VSTX)
        :param doc_name: visio file/document name(type: str) -> it can be existed and non-existed
        :param app: visio app(type: IVApplication)
        :return: document(type: IVDocument)
        """
        if not app and self.app is not None:
            doc = self.app.Documents.Add(doc_name)
        else:
            doc = app.Documents.Add(doc_name)

        return doc

    def add_page(self, doc, number):
        """
        add page in exist document
        :param doc: document(you create before; type: IVDocument)
        :param number: page num(type: int)
        :return: page(type: IVPage)
        """
        page = doc.Pages.Item(number)

        return page

    def choose_shape(self, shape_name, template):
        """
        get shape template in template file/document
        :param shape_name: shape template name(type: str)
        :param template: template file/document(you read in before; type: IVDocument)
        :return: shape template(type: IVMaster)
        """
        master = template.Masters.ItemU(shape_name)

        return master

    def add_shape(self, page, master, x, y):
        """
        add shape(template) in existing page via center point(x, y)
        :param page: page you add before(type:IVPage)
        :param master: shape template(type: IVMaster)
        :param x: X-axis of center point(type: int/str) -> warning: can't support type likes:'2 pt'
        :param y: Y-axis of center point(type:int/str)
        :return: shape(add in page; type:IVShape)
        x: column (1 is for 25pt in visio)
        y: row
        """
        shp = page.Drop(master, x, y)

        return shp

    def resize_shape(self, shp, w, h):
        """
        resize shape
        :param shp: shape (type: IVShape)
        :param w: resize width(type: str/int) -> 1 = '25 pt'
        :param h: resize heigth(type: str/int) -> 1 = '25 pt'
        :return: None
        """
        shp.Cells('Width').FormulaU = w
        shp.Cells('Height').FormulaU = h

    def shape_text(self, shp, text):
        """
        set shape text content
        :param shp: shape (type: IVShape)
        :param text: text content(type: str)
        :return:None
        """
        shp.Text = text

    def shape_fillstyle(self, shp, style):
        """
        set shape fillstyle
        :param shp: shape (type: IVShape)
        :param style: fill style(type: str) -> ('None': non-fill; 'Normal': normal fill)
        :return:None
        """
        shp.FillStyle = style

    def set_attr(self, shp, attr_name, attr_val):
        """
        set shape attribute
        :param shp: shape (type:IVShape)
        :param attr_name: (type:str)
        :param attr_val: (type:str/int)
        :return:None
        example:
            set shape line_color:       serv.Cells("LineColor").FormulaU = 0 (0: black; 1: white ...) | 'RGB(255,0,0)'
            set shape line_weight:      serv.Cells('LineWeight').FormulaU = '2.0 pt' | 2 (1 = 25 pt)
            set shape text_size:        serv.Cells("Char.size").FormulaU = "20 pt"
            set shape fill_color:       serv.Cells('Fillforegnd').FormulaForceU = 'RGB(255,0,0)'
        """
        shp.Cells(attr_name).FormulaU = attr_val

    def auto_connect(self, shp1, shp2, style, connect):
        """
        (auto) connect shp1 with shp2 via style
        :param shp1: shape(connect begin; type:IVShape)
        :param shp2: shape(connect end; type:IVShape)
        :param style: connect style(four choice: down, up, right, left)
        :param connect: connect shape
        """
        connect_style = {
            'down': constants.visAutoConnectDirDown,
            'up': constants.visAutoConnectDirUp,
            'right': constants.visAutoConnectDirRight,
            'left': constants.visAutoConnectDirLeft
        }
        if not connect:
            shp1.AutoConnect(shp2, connect_style[style])
        else:
            shp1.AutoConnect(shp2, connect_style[style], connect)

    # TODO: 1. Connect
    def connect(self, page, shp1, shp2, shp1_point, shp2_point, app=None):
        """
        connect shp1 with shp2 via points
        :param app: visio app(type: IVApplication)
        :param page: page you add before(type:IVPage)
        :param shp1: shape(connect begin; type:IVShape)
        :param shp2: shape(connect end; type:IVShape)
        :param shp1_point: shape(begin) connect point(type: int)
        :param shp2_point: shape(end) connect point(type: int)
        example: rectangle has 5 shape points:
            0 -> center
            1 -> down
            2 -> right
            3 -> up
            4 -> left
        more shape points see documents
        """
        shape_points = {
            0: "Connections.X1",
            1: "Connections.X2",
            2: "Connections.X3",
            3: "Connections.X4",
            4: "Connections.X5",
            5: "Connections.X6",
            6: "Connections.X7",
            7: "Connections.X8",
            8: "Connections.X9",
        }
        if not app:
            app = self.app
        conn = self.add_shape(page=page, master=app.ConnectorToolDataObject, x=0, y=0)
        conn_begin = conn.Cells('BeginX')
        conn_end = conn.Cells('EndX')
        vsoCellGlueToObjectshp1 = shp1.Cells(shape_points[shp1_point])
        vsoCellGlueToObjectshp2 = shp2.Cells(shape_points[shp2_point])
        conn_begin.GlueTo(vsoCellGlueToObjectshp1)
        conn_end.GlueTo(vsoCellGlueToObjectshp2)

    def save(self, doc, file_path):
        """
        save visio file/document as file_path
        :param doc: visio file/document you create before(type:IVDocument)
        :param file_path: save path(type: str) -> it can supprot relative path and absolute path
        :return:None
        """
        doc.SaveAs(file_path)
        print('save succeed!')

    def shape_contract(self, template):
        """
        get all shape template name(the version language of Visio that you installed: English) in template
        :param template: template file/document(you read in before; type: IVDocument)
        :return: dict{Name: NameU}
        """
        elements_name_dict = {}
        for elem in template.Masters:
            elements_name_dict[elem.Name] = elem.NameU
        return elements_name_dict

    def close(self, doc, app):
        doc.Close()
        app.Quit()
        print('close succeed!')