// Copyright 2026 The ODML Authors.
  
// Licensed under the Apache License, Version 2.0 (the "License");
                                                                   
                                          
  
                                                  
  
                                                                      
                                                                    
                                                                           
                                                                      
                                 

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_API_EXPORT_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_API_EXPORT_H_

                                                                            
                                    
  
                                                                               
                                                                                                       
  
                                                                                
                                              
                                                                       
  
                                                                             
                                                                         
                                                                               
                      
#ifndef LITERT_LM_C_API_EXPORT
#if defined(_WIN32)
#define LITERT_LM_C_API_EXPORT __declspec(dllexport)
#else
                                                                    
                       
#define LITERT_LM_C_API_EXPORT __attribute__((visibility("default")))
#endif                    
#endif                           

#endif                                               
