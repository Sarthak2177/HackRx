import re
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

class DocumentPatternAnalyzer:
    def __init__(self):
        self.age_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*age|old)',
            r'age\s*(?:limit|requirement|criteria)[\s:]*(\d+)',
            r'minimum\s*age[\s:]*(\d+)',
            r'maximum\s*age[\s:]*(\d+)',
            r'senior\s*citizen[\s:]*(\d+)',
            r'adult[\s:]*(\d+)'
        ]
        
        self.coverage_patterns = [
            r'(?:covered|covers|coverage)[\s:]*([^.]+)',
            r'(?:eligible|eligibility)[\s:]*([^.]+)',
            r'(?:excluded|exclusions?)[\s:]*([^.]+)',
            r'(?:benefits?)[\s:]*([^.]+)'
        ]
        
        self.condition_patterns = [
            r'(?:conditions?|requirements?)[\s:]*([^.]+)',
            r'(?:subject\s*to|provided\s*that)[\s:]*([^.]+)',
            r'(?:waiting\s*period)[\s:]*([^.]+)',
            r'(?:pre-?authorization)[\s:]*([^.]+)'
        ]
        
        self.location_patterns = [
            r'(?:in|at|within)\s*([A-Za-z\s]+(?:city|state|region|area))',
            r'(?:network\s*hospitals?|empanelled\s*hospitals?)[\s:]*([^.]+)',
            r'(?:location|region|zone)[\s:]*([^.]+)'
        ]
    
    def extract_rules_from_documents(self, chunks: List[str]) -> Dict[str, Any]:
        """Dynamically extract business rules from document chunks"""
        
        combined_text = " ".join(chunks)
        
        extracted_rules = {
            'age_rules': self._extract_age_rules(combined_text),
            'coverage_rules': self._extract_coverage_rules(combined_text),
            'condition_rules': self._extract_condition_rules(combined_text),
            'location_rules': self._extract_location_rules(combined_text),
            'eligibility_criteria': self._extract_eligibility_criteria(combined_text)
        }
        
        return extracted_rules
    
    def _extract_age_rules(self, text: str) -> Dict[str, Any]:
        """Extract age-related rules from text"""
        age_rules = {
            'minimum_ages': [],
            'maximum_ages': [],
            'special_categories': {}
        }
        
        for pattern in self.age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                age = int(match.group(1))
                context = text[max(0, match.start()-50):match.end()+50]
                
                if 'minimum' in context.lower() or 'above' in context.lower():
                    age_rules['minimum_ages'].append((age, context.strip()))
                elif 'maximum' in context.lower() or 'below' in context.lower():
                    age_rules['maximum_ages'].append((age, context.strip()))
                elif 'senior' in context.lower():
                    age_rules['special_categories']['senior_citizen'] = age
                
        return age_rules
    
    def _extract_coverage_rules(self, text: str) -> Dict[str, List[str]]:
        """Extract coverage-related rules"""
        coverage_rules = {
            'covered_items': [],
            'excluded_items': [],
            'benefits': []
        }
        
        # Find coverage mentions
        coverage_matches = re.finditer(r'(?:covered|covers|coverage|includes?)[\s:]*([^.!?]+)', text, re.IGNORECASE)
        for match in coverage_matches:
            item = match.group(1).strip()
            if item and len(item) > 10:
                coverage_rules['covered_items'].append(item)
        
        # Find exclusions
        exclusion_matches = re.finditer(r'(?:excluded?|not\s*covered|except)[\s:]*([^.!?]+)', text, re.IGNORECASE)
        for match in exclusion_matches:
            item = match.group(1).strip()
            if item and len(item) > 10:
                coverage_rules['excluded_items'].append(item)
        
        return coverage_rules
    
    def _extract_condition_rules(self, text: str) -> List[str]:
        """Extract conditions and requirements"""
        conditions = []
        
        for pattern in self.condition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip()
                if condition and len(condition) > 10:
                    conditions.append(condition)
        
        return list(set(conditions))  # Remove duplicates
    
    def _extract_location_rules(self, text: str) -> Dict[str, List[str]]:
        """Extract location-specific information"""
        location_rules = {
            'network_hospitals': [],
            'service_areas': [],
            'regional_conditions': []
        }
        
        # Extract hospital networks
        hospital_matches = re.finditer(r'(?:hospital|medical\s*center|clinic)[\s:]*([^.!?]+)', text, re.IGNORECASE)
        for match in hospital_matches:
            hospital = match.group(1).strip()
            if hospital and len(hospital) > 5:
                location_rules['network_hospitals'].append(hospital)
        
        return location_rules
    
    def _extract_eligibility_criteria(self, text: str) -> List[str]:
        """Extract eligibility criteria"""
        criteria = []
        
        # Look for eligibility-related sentences
        eligibility_matches = re.finditer(r'(?:eligible|qualify|qualification|criteria)[\s:]*([^.!?]+)', text, re.IGNORECASE)
        for match in eligibility_matches:
            criterion = match.group(1).strip()
            if criterion and len(criterion) > 10:
                criteria.append(criterion)
        
        return list(set(criteria))
