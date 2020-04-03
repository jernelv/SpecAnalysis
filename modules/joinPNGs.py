import PIL # Image
class moduleClass:
    filetypes = ['png']
    def __init__(self, fig, locations, frame, ui):
        images=[]
        for location in locations:
            images.append(PIL.Image.open(location))
        widths, heights = zip(*(i.size for i in images))
        if ui['joinPNG_orientation']=='horizontal':
            total_width = sum(widths)
            max_height = max(heights)
        else:# ui['joinPNG_orientation']=='vertical':
            total_width = max(widths)
            max_height = sum(heights)

        new_im = PIL.Image.new('RGB', (total_width, max_height))

        offset = 0
        if ui['joinPNG_orientation']=='horizontal':
            for im in images:
                new_im.paste(im, (offset,0))
                offset += im.size[0]
        else:# ui['joinPNG_orientation']=='vertical':
            for im in images:
                new_im.paste(im, (0,offset))
                offset += im.size[1]
        new_im.save(locations[0].strip('.png')+'_combined.png')
    def addButtons():
        buttons = [
		{'key': 'joinPNG_orientation', 'type': 'radio:text', 'texts': ['horizontal', 'vertical'], 'tab': 0, 'row': 0} ,
        ]
        return buttons
