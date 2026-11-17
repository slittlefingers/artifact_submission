








                                                                                   
                                                                                     
                      
TB_SIZE = 0x88                                 
TB_AHWB = 0xa0                                               
TB_LEN = 0xb0                                
                       
GB_NH = 0x70                            
GB_LEN = 0x78
                                                                                                 
                                                                                                      
                                                                                                   
                                                                        
SSO_KEY_BACK = 1                                                             
SSO_TB = 0x20                                             
                                                                     
NH_VER = 0                                   
NH_NUMFDS = 4
NH_FD = 12                                          
NH_LEN = 20
NH_VERSION_MAGIC = 12
                                                                     
HEAP_LO, HEAP_HI, BANKS = 0x005000000000, 0x008000000000, 4

                                                                  
HOST_ELEM = 0x08                                                
HOST_BT = 0x80                                          
HOST_SIZE = 0x88                      
HOST_ADDR = 0xa0                                                    
HOST_HDR = 0xb0                                       
                                                                                                  
HOST_SIG = bytes.fromhex("09000000" "04000000" "01000000" "01000000")
MTE_MASK = (1 << 40) - 1                                                        

                                                                            
MARK_PROTECTED = b"TENSOR PROTECTED"                                    
MARK_FAILREAD = b"LEMON FAIL READ "                                                

                          
LIME_MAGIC = 0x4C694D45
LIME_HDR = 32
PAGE = 4096
