// Copyright 2026 The ODML Authors.
  
// Licensed under the Apache License, Version 2.0 (the "License");
                                                                   
                                          
  
                                                  
  
                                                                      
                                                                    
                                                                           
                                                                      
                                 

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ERROR_REPORTER_H_

#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"           
#else
#include "c/api_export.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

                                                                                
                                       
                                                                                
  
                                                                           
                           
  
                      
                                                                              
                                                                             
                                                               
  
                                    
                                                                           
                                                                           
  
                                                 
                                                                            
                                                                             
               
                                                                             
                                                                               
                                                                     
                                                                            
                                                                             
                         
  
                                    
                                                                               
                                                                    
                                                                            
                                                                             
                                                                              
                                               
                                                                                
                                                                        
                      
  
                                                  
                                                                          
                                              
                                                                           
                                                                        
                                                                 
                                                                      
                                                                            
                                                                            
                                                                
  
                                           
                                                                    
                                                                         
                                                    
                                                                            
                                                
                                                                        
                                                                         
                                                                         
                                                  
                                                                              
                                                                              
                                                    
                                                                        
                                                                  
                                             
                                                                        
                                                                         
                                
  
              
  
                                      
                                                                               
                                          
                                                               
                                                                    
                                  
                        
         
                                                                                

                                                                                
               
                                                                                
  
                                                                
  
                                                                 
                                            
  
                 
                                                                      
                 
                                                                       
                                                                        
                                                                                
                                                                                
                                                            
                                                                           
                                                                              
                                                                               
  
                                                          
                                                                                
                                                                              
                                                                              
                                                                            
                                            
  
                          
typedef enum LiteRtLmStatusCode {
  kLiteRtLmStatusOk = 0,
  kLiteRtLmStatusCancelled = 1,
  kLiteRtLmStatusUnknown = 2,
  kLiteRtLmStatusInvalidArgument = 3,
  kLiteRtLmStatusDeadlineExceeded = 4,
  kLiteRtLmStatusNotFound = 5,
  kLiteRtLmStatusAlreadyExists = 6,
  kLiteRtLmStatusPermissionDenied = 7,
  kLiteRtLmStatusResourceExhausted = 8,
  kLiteRtLmStatusFailedPrecondition = 9,
  kLiteRtLmStatusAborted = 10,
  kLiteRtLmStatusOutOfRange = 11,
  kLiteRtLmStatusUnimplemented = 12,
  kLiteRtLmStatusInternal = 13,
  kLiteRtLmStatusUnavailable = 14,
  kLiteRtLmStatusDataLoss = 15,
  kLiteRtLmStatusUnauthenticated = 16,
} LiteRtLmStatusCode;

                                                                 
  
                                          
                                                                            
                                                                     
                                                                           
                                                                              
                                                                   
  
                   
                                                                            
                                                              
  
                               
                                                                     
                                                                             
                                                                               
                                                                               
                                                                            
                                                                              
  
           
                                                                              
                                                                             
           
  
                          
LITERT_LM_C_API_EXPORT
const char* litert_lm_get_last_error_message(void);

                                                                            
                   
  
                                 
                                                                         
                                                                                
                                                                            
                                                                             
                                                                           
                                
  
                   
                                                                            
                             
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_clear_last_error(void);

#ifdef __cplusplus
}               
#endif

#endif                                                   
