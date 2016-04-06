#from onvif.client import ONVIFService
#from onvif.client import ONVIFCamera
#from onvif.client import SERVICES
#from onvif.exceptions import ONVIFError
#from onvif.exceptions import ERR_ONVIF_UNKNOWN
#from onvif.exceptions import ERR_ONVIF_PROTOCOL
#from onvif.exceptions import ERR_ONVIF_WSDL
#from onvif.exceptions import ERR_ONVIF_BUILD
#from onvif import cli
#
#__all__ = ( 'ONVIFService',
#            'ONVIFCamera',
#            'ONVIFError',
#            'ERR_ONVIF_UNKNOWN',
#            'ERR_ONVIF_PROTOCOL',
#            'ERR_ONVIF_WSDL',
#            'ERR_ONVIF_BUILD',
#            'SERVICES',
#            'cli')

# comente lo anteriorior porque me daba:
# ImportError: No module named onvif.client
# Asi que hice un __init__ tradicional
# seba - 5/04/2016
__all__ = [ 'cli',
            'client',
            'definition',
            'exceptions']
