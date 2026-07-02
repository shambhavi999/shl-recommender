"""
scripts/scrape_catalog.py

Scrapes the SHL Product Catalog restricted to Individual Test Solutions.
Falls back to an embedded catalog if the website blocks scraping.

Usage:
    python scripts/scrape_catalog.py --out data/catalog.json
    python scripts/scrape_catalog.py --out data/catalog.json --use-embedded
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
# Correct URL (with /solutions/)
LISTING_URL = f"{BASE}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED CATALOG — full list of SHL Individual Test Solutions
# Used as fallback when website scraping is blocked/unavailable.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDED_CATALOG = [
  # ── Ability & Aptitude ────────────────────────────────────────────────────
  {"name":"Verify G+","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/","test_type":["A"],"description":"Measures general cognitive ability including numerical, verbal and inductive reasoning. Suitable for all professional and managerial roles.","job_levels":["Graduate","Professional Individual Contributor","Manager","Director"]},
  {"name":"Verify Numerical Reasoning","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-reasoning-test/","test_type":["A"],"description":"Measures the ability to understand and apply numerical concepts and data presented in charts and tables.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Verify Verbal Reasoning","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-reasoning-test/","test_type":["A"],"description":"Measures the ability to evaluate and draw inferences from written passages and verbal information.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Verify Inductive Reasoning","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-inductive-reasoning-test/","test_type":["A"],"description":"Measures ability to identify patterns and relationships in abstract visual information.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Verify Deductive Reasoning","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-deductive-reasoning-test/","test_type":["A"],"description":"Measures ability to draw logical conclusions from given information and rules.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Verify Checking","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-checking/","test_type":["A"],"description":"Measures speed and accuracy of checking and comparing information.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Verify Calculation","url":"https://www.shl.com/solutions/products/product-catalog/view/verify-calculation/","test_type":["A"],"description":"Measures basic numerical calculation skills for clerical and administrative roles.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Numerical Reasoning - Graduate","url":"https://www.shl.com/solutions/products/product-catalog/view/numerical-reasoning-graduate/","test_type":["A"],"description":"Graduate-level numerical reasoning assessment.","job_levels":["Graduate"]},
  {"name":"Verbal Reasoning - Graduate","url":"https://www.shl.com/solutions/products/product-catalog/view/verbal-reasoning-graduate/","test_type":["A"],"description":"Graduate-level verbal reasoning assessment.","job_levels":["Graduate"]},
  {"name":"Management and Graduate Item Bank (MGIB) - Numerical","url":"https://www.shl.com/solutions/products/product-catalog/view/mgib-numerical/","test_type":["A"],"description":"Advanced numerical reasoning for management and graduate roles.","job_levels":["Graduate","Manager"]},
  {"name":"Management and Graduate Item Bank (MGIB) - Verbal","url":"https://www.shl.com/solutions/products/product-catalog/view/mgib-verbal/","test_type":["A"],"description":"Advanced verbal reasoning for management and graduate roles.","job_levels":["Graduate","Manager"]},
  {"name":"Critical Reasoning Test Battery (CRTB2) - Numerical","url":"https://www.shl.com/solutions/products/product-catalog/view/crtb2-numerical/","test_type":["A"],"description":"Critical numerical reasoning for professional and managerial candidates.","job_levels":["Professional Individual Contributor","Manager","Director"]},
  {"name":"Critical Reasoning Test Battery (CRTB2) - Verbal","url":"https://www.shl.com/solutions/products/product-catalog/view/crtb2-verbal/","test_type":["A"],"description":"Critical verbal reasoning for professional and managerial candidates.","job_levels":["Professional Individual Contributor","Manager","Director"]},
  # ── Personality & Behavior ────────────────────────────────────────────────
  {"name":"OPQ32r","url":"https://www.shl.com/solutions/products/product-catalog/view/opq32r/","test_type":["P"],"description":"Occupational Personality Questionnaire normative version. Measures 32 personality characteristics relevant to work performance including relationships, thinking style and feelings and emotions.","job_levels":["Graduate","Professional Individual Contributor","Manager","Director","Executive"]},
  {"name":"OPQ32n","url":"https://www.shl.com/solutions/products/product-catalog/view/opq32n/","test_type":["P"],"description":"Occupational Personality Questionnaire normative version measuring 32 workplace personality characteristics.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"MQ (Motivational Questionnaire)","url":"https://www.shl.com/solutions/products/product-catalog/view/mq/","test_type":["P"],"description":"Measures motivational factors that energise and drive individual performance at work. Assesses energy and dynamism, synergy, and intrinsic motivation.","job_levels":["Professional Individual Contributor","Manager","Director","Executive"]},
  {"name":"AI Skills","url":"https://www.shl.com/solutions/products/product-catalog/view/ai-skills/","test_type":["P"],"description":"Assesses an individual's readiness and aptitude for working with AI tools and technologies in the workplace.","job_levels":["Entry-Level","Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Global Skills Development Report","url":"https://www.shl.com/solutions/products/product-catalog/view/global-skills-development-report/","test_type":["A","E","B","C","D","P"],"description":"Comprehensive development report combining multiple assessment types for talent development.","job_levels":["Professional Individual Contributor","Manager","Director"]},
  {"name":"Work Strengths","url":"https://www.shl.com/solutions/products/product-catalog/view/work-strengths/","test_type":["P"],"description":"Identifies individual strengths relevant to workplace performance and engagement.","job_levels":["Entry-Level","Graduate","Professional Individual Contributor"]},
  # ── Biodata & Situational Judgement ──────────────────────────────────────
  {"name":"Customer Contact Styles Questionnaire (CCSQ)","url":"https://www.shl.com/solutions/products/product-catalog/view/ccsq/","test_type":["B"],"description":"Situational judgement test measuring preferred approaches to customer service interactions.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Graduate Situational Judgement Test","url":"https://www.shl.com/solutions/products/product-catalog/view/graduate-sjt/","test_type":["B"],"description":"Situational judgement test assessing graduate candidates on workplace scenarios.","job_levels":["Graduate"]},
  {"name":"Manager Situational Judgement Test","url":"https://www.shl.com/solutions/products/product-catalog/view/manager-sjt/","test_type":["B"],"description":"Situational judgement test measuring managerial decision-making and people management effectiveness.","job_levels":["Manager","Front Line Manager","Supervisor"]},
  {"name":"Sales Situational Judgement Test","url":"https://www.shl.com/solutions/products/product-catalog/view/sales-sjt/","test_type":["B"],"description":"Situational judgement test for sales roles measuring customer interaction and deal management skills.","job_levels":["Entry-Level","Graduate","Professional Individual Contributor"]},
  # ── Simulations ───────────────────────────────────────────────────────────
  {"name":"Automata (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/automata-new/","test_type":["S"],"description":"Live coding simulation that presents real-world programming tasks and measures code quality and problem solving.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Automata - Fix (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/","test_type":["S"],"description":"Live coding simulation where candidates fix broken code, measuring debugging skills and code comprehension.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Automata - SQL (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/automata-sql-new/","test_type":["S"],"description":"Live SQL coding simulation measuring database query writing skills in a realistic environment.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Accounts Payable Simulation (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/accounts-payable-simulation-new/","test_type":["S"],"description":"Simulation measuring accounts payable processing skills and accuracy in a realistic work environment.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Accounts Receivable Simulation (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/accounts-receivable-simulation-new/","test_type":["S"],"description":"Simulation measuring accounts receivable skills including invoice processing and collections.","job_levels":["Entry-Level","Graduate"]},
  # ── Assessment Exercises ──────────────────────────────────────────────────
  {"name":"Assessment and Development Center Exercises","url":"https://www.shl.com/solutions/products/product-catalog/view/assessment-and-development-center-exercises/","test_type":["E"],"description":"Suite of assessment center exercises including inbox exercises, role plays, group discussions and presentations.","job_levels":["Manager","Director","Executive"]},
  # ── Knowledge & Skills — Programming Languages ────────────────────────────
  {"name":"Java 8 (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/java-8-new/","test_type":["K"],"description":"Multi-choice test measuring knowledge of Java 8 features including lambdas, streams, generics, collections and concurrency.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Java (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/java-new/","test_type":["K"],"description":"Multi-choice test measuring core Java programming knowledge including OOP, data structures and design patterns.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Python (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/python-new/","test_type":["K"],"description":"Multi-choice test measuring Python programming knowledge including data structures, OOP, libraries and scripting.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"SQL (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/sql-new/","test_type":["K"],"description":"Multi-choice test measuring SQL query writing, data manipulation, joins, subqueries and transaction processing.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"C++ (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/c-plus-plus-new/","test_type":["K"],"description":"Multi-choice test measuring C++ programming knowledge including memory management, OOP and STL.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"C# (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/c-sharp-new/","test_type":["K"],"description":"Multi-choice test measuring C# programming knowledge including .NET framework, OOP and LINQ.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"JavaScript (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/javascript-new/","test_type":["K"],"description":"Multi-choice test measuring JavaScript knowledge including ES6+, DOM manipulation, async programming and closures.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"PHP (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/php-new/","test_type":["K"],"description":"Multi-choice test measuring PHP programming knowledge including web development and server-side scripting.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Ruby (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/ruby-new/","test_type":["K"],"description":"Multi-choice test measuring Ruby programming knowledge including Rails framework and OOP concepts.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Scala (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/scala-new/","test_type":["K"],"description":"Multi-choice test measuring Scala programming knowledge including functional programming and Akka.","job_levels":["Professional Individual Contributor"]},
  {"name":"Swift (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/swift-new/","test_type":["K"],"description":"Multi-choice test measuring Swift programming knowledge for iOS and macOS development.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Kotlin (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/kotlin-new/","test_type":["K"],"description":"Multi-choice test measuring Kotlin programming knowledge including Android development.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Go (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/go-new/","test_type":["K"],"description":"Multi-choice test measuring Go (Golang) programming knowledge including concurrency and microservices.","job_levels":["Professional Individual Contributor"]},
  {"name":"R (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/r-new/","test_type":["K"],"description":"Multi-choice test measuring R programming knowledge for statistical computing and data analysis.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Web & Frameworks ────────────────────────────────
  {"name":"Angular 6 (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/angular-6-new/","test_type":["K"],"description":"Multi-choice test measuring Angular 6 framework knowledge including components, services and RxJS.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"AngularJS (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/angularjs-new/","test_type":["K"],"description":"Multi-choice test measuring AngularJS framework knowledge for front-end web development.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"React (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/react-new/","test_type":["K"],"description":"Multi-choice test measuring React.js knowledge including hooks, state management and component lifecycle.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Spring (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/spring-new/","test_type":["K"],"description":"Multi-choice test measuring Spring framework knowledge including Spring Core, AOP, IOC container and transactions.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Node.js (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/node-js-new/","test_type":["K"],"description":"Multi-choice test measuring Node.js knowledge including event loop, async programming and Express.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"ASP.NET with C# (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/asp-net-with-c-new/","test_type":["K"],"description":"Multi-choice test measuring ASP.NET with C# knowledge for web application development.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":".NET Framework 4.5","url":"https://www.shl.com/solutions/products/product-catalog/view/net-framework-4-5/","test_type":["K"],"description":"Multi-choice test measuring .NET Framework 4.5 knowledge.","job_levels":["Professional Individual Contributor"]},
  # ── Knowledge & Skills — Cloud & DevOps ──────────────────────────────────
  {"name":"Amazon Web Services (AWS) Development (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/amazon-web-services-aws-development-new/","test_type":["K"],"description":"Multi-choice test measuring AWS cloud development knowledge including EC2, S3, Lambda and core services.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Microsoft Azure (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/microsoft-azure-new/","test_type":["K"],"description":"Multi-choice test measuring Microsoft Azure cloud services knowledge.","job_levels":["Professional Individual Contributor"]},
  {"name":"Google Cloud Platform (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/google-cloud-platform-new/","test_type":["K"],"description":"Multi-choice test measuring Google Cloud Platform services and development knowledge.","job_levels":["Professional Individual Contributor"]},
  {"name":"Agile Software Development","url":"https://www.shl.com/solutions/products/product-catalog/view/agile-software-development/","test_type":["K"],"description":"Multi-choice test measuring Agile methodology knowledge including Scrum, Kanban and XP practices.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Agile Testing (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/agile-testing-new/","test_type":["K"],"description":"Multi-choice test measuring agile testing practices and methodologies.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"DevOps (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/devops-new/","test_type":["K"],"description":"Multi-choice test measuring DevOps practices including CI/CD, containerisation and automation.","job_levels":["Professional Individual Contributor"]},
  {"name":"Docker (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/docker-new/","test_type":["K"],"description":"Multi-choice test measuring Docker containerisation knowledge.","job_levels":["Professional Individual Contributor"]},
  {"name":"Kubernetes (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/kubernetes-new/","test_type":["K"],"description":"Multi-choice test measuring Kubernetes container orchestration knowledge.","job_levels":["Professional Individual Contributor"]},
  # ── Knowledge & Skills — Data & Big Data ─────────────────────────────────
  {"name":"Apache Hadoop (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/apache-hadoop-new/","test_type":["K"],"description":"Multi-choice test measuring Apache Hadoop knowledge for big data processing.","job_levels":["Professional Individual Contributor"]},
  {"name":"Apache Spark (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/apache-spark-new/","test_type":["K"],"description":"Multi-choice test measuring Apache Spark knowledge for large-scale data processing.","job_levels":["Professional Individual Contributor"]},
  {"name":"Apache Kafka (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/apache-kafka-new/","test_type":["K"],"description":"Multi-choice test measuring Apache Kafka knowledge for real-time data streaming.","job_levels":["Professional Individual Contributor"]},
  {"name":"Machine Learning (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/machine-learning-new/","test_type":["K"],"description":"Multi-choice test measuring machine learning concepts, algorithms and application knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Data Science (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/data-science-new/","test_type":["K"],"description":"Multi-choice test measuring data science knowledge including statistics, modelling and analysis.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Statistics (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/statistics-new/","test_type":["K"],"description":"Multi-choice test measuring statistical knowledge and data analysis techniques.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Testing & QA ────────────────────────────────────
  {"name":"Manual Testing (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/manual-testing-new/","test_type":["K"],"description":"Multi-choice test measuring software testing lifecycle, test case design, tools and techniques.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Selenium (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/selenium-new/","test_type":["K"],"description":"Multi-choice test measuring Selenium WebDriver automation testing knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Databases ───────────────────────────────────────
  {"name":"MySQL (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/mysql-new/","test_type":["K"],"description":"Multi-choice test measuring MySQL database administration and query knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Oracle SQL (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/oracle-sql-new/","test_type":["K"],"description":"Multi-choice test measuring Oracle SQL and PL/SQL knowledge.","job_levels":["Professional Individual Contributor"]},
  {"name":"MongoDB (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/mongodb-new/","test_type":["K"],"description":"Multi-choice test measuring MongoDB NoSQL database knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Mobile ───────────────────────────────────────────
  {"name":"Android Development (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/android-development-new/","test_type":["K"],"description":"Multi-choice test measuring Android app development knowledge including Java/Kotlin and Android SDK.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"iOS Development (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/ios-development-new/","test_type":["K"],"description":"Multi-choice test measuring iOS app development knowledge including Swift and UIKit.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Networking & Security ───────────────────────────
  {"name":"Networking (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/networking-new/","test_type":["K"],"description":"Multi-choice test measuring computer networking knowledge including protocols, routing and security.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Cyber Security (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/cyber-security-new/","test_type":["K"],"description":"Multi-choice test measuring cyber security knowledge including threats, vulnerabilities and protection strategies.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Finance & Accounting ────────────────────────────
  {"name":"Accounts Payable (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/accounts-payable-new/","test_type":["K"],"description":"Multi-choice test measuring accounts payable knowledge including invoice processing and vendor management.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Accounts Receivable (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/accounts-receivable-new/","test_type":["K"],"description":"Multi-choice test measuring accounts receivable knowledge including collections and cash application.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Financial Accounting (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/financial-accounting-new/","test_type":["K"],"description":"Multi-choice test measuring financial accounting principles, standards and reporting.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Management Accounting (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/management-accounting-new/","test_type":["K"],"description":"Multi-choice test measuring management accounting knowledge including budgeting, costing and variance analysis.","job_levels":["Professional Individual Contributor","Manager"]},
  # ── Knowledge & Skills — Business & Management ───────────────────────────
  {"name":"Human Resources (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/human-resources-new/","test_type":["K"],"description":"Multi-choice test measuring HR management knowledge including recruitment, training, performance management and compensation.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Marketing (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/marketing-new/","test_type":["K"],"description":"Multi-choice test measuring marketing knowledge including market research, brand management, consumer behaviour and digital marketing.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Sales (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/sales-new/","test_type":["K"],"description":"Multi-choice test measuring sales knowledge including techniques, CRM and pipeline management.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Project Management (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/project-management-new/","test_type":["K"],"description":"Multi-choice test measuring project management knowledge including planning, risk, scope and stakeholder management.","job_levels":["Graduate","Professional Individual Contributor","Manager"]},
  {"name":"Supply Chain Management (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/supply-chain-management-new/","test_type":["K"],"description":"Multi-choice test measuring supply chain and logistics knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Customer Service (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/customer-service-new/","test_type":["K"],"description":"Multi-choice test measuring customer service skills and knowledge.","job_levels":["Entry-Level","Graduate"]},
  # ── Knowledge & Skills — Engineering ─────────────────────────────────────
  {"name":"Mechanical Engineering (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/mechanical-engineering-new/","test_type":["K"],"description":"Multi-choice test measuring mechanical engineering concepts, thermodynamics and materials science.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Electrical Engineering (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/electrical-engineering-new/","test_type":["K"],"description":"Multi-choice test measuring electrical engineering knowledge including circuits, power systems and electronics.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Civil Engineering (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/civil-engineering-new/","test_type":["K"],"description":"Multi-choice test measuring civil engineering knowledge including structures, materials and construction.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Aeronautical Engineering (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/aeronautical-engineering-new/","test_type":["K"],"description":"Multi-choice test measuring aeronautical engineering knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  {"name":"Aerospace Engineering (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/aerospace-engineering-new/","test_type":["K"],"description":"Multi-choice test measuring aerospace engineering knowledge.","job_levels":["Graduate","Professional Individual Contributor"]},
  # ── Knowledge & Skills — Microsoft Office ────────────────────────────────
  {"name":"Microsoft Excel (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-new/","test_type":["K"],"description":"Multi-choice test measuring Microsoft Excel knowledge including formulas, pivot tables and data analysis.","job_levels":["Entry-Level","Graduate","Professional Individual Contributor"]},
  {"name":"Microsoft Word (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/microsoft-word-new/","test_type":["K"],"description":"Multi-choice test measuring Microsoft Word knowledge for document creation and formatting.","job_levels":["Entry-Level","Graduate"]},
  {"name":"Microsoft PowerPoint (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/microsoft-powerpoint-new/","test_type":["K"],"description":"Multi-choice test measuring Microsoft PowerPoint knowledge for presentation creation.","job_levels":["Entry-Level","Graduate"]},
]


def fetch_with_retry(url, session, delay=1.0, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(delay)
            return resp.text
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_listing_page(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    # Try multiple strategies to find the assessments table
    # Strategy 1: Look for table with 'Individual Test Solutions' header
    for table in soup.find_all("table"):
        header_text = table.get_text()
        if "Individual Test Solutions" not in header_text:
            continue
        for tr in table.find_all("tr"):
            link = tr.find("a", href=True)
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link["href"]
            url = href if href.startswith("http") else BASE + href
            tds = tr.find_all("td")
            test_type = []
            if tds:
                letters_text = tds[-1].get_text(separator=" ", strip=True)
                test_type = [c for c in letters_text.split() if c in TEST_TYPE_LABELS]
            if name:
                rows.append({"name": name, "url": url, "test_type": test_type})

    # Strategy 2: look for any product-catalog links
    if not rows:
        for a in soup.find_all("a", href=True):
            if "/product-catalog/view/" in a["href"]:
                name = a.get_text(strip=True)
                href = a["href"]
                url = href if href.startswith("http") else BASE + href
                if name:
                    rows.append({"name": name, "url": url, "test_type": []})
    return rows


def scrape_live(out_path, delay, limit):
    session = requests.Session()
    urls_to_try = [
        f"{LISTING_URL}?type=1",
        f"{LISTING_URL}?type=1&start=0",
        LISTING_URL,
    ]
    all_rows = []
    for url in urls_to_try:
        print(f"Trying: {url}", file=sys.stderr)
        html = fetch_with_retry(url, session, delay=delay)
        if html:
            rows = parse_listing_page(html)
            if rows:
                all_rows.extend(rows)
                print(f"  Found {len(rows)} items from {url}", file=sys.stderr)
                # Try pagination
                for start in range(12, 500, 12):
                    page_url = f"{LISTING_URL}?type=1&start={start}"
                    ph = fetch_with_retry(page_url, session, delay=delay)
                    if not ph:
                        break
                    page_rows = parse_listing_page(ph)
                    if not page_rows:
                        break
                    all_rows.extend(page_rows)
                    print(f"  Page start={start}: {len(page_rows)} more items", file=sys.stderr)
                break

    # Deduplicate
    seen, deduped = set(), []
    for r in all_rows:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)
    return deduped[:limit] if limit else deduped


def save_catalog(catalog, out_path):
    for i, a in enumerate(catalog, start=1):
        a.setdefault("id", f"shl-{i:04d}")
        a.setdefault("test_type", [])
        a.setdefault("description", "")
        a.setdefault("job_levels", [])
        a.setdefault("languages", [])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(catalog)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/catalog.json")
    parser.add_argument("--use-embedded", action="store_true",
                        help="Skip live scraping; write the embedded catalog directly")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.8)
    args = parser.parse_args()

    if args.use_embedded:
        print(f"Using embedded catalog ({len(EMBEDDED_CATALOG)} items)", file=sys.stderr)
        save_catalog(EMBEDDED_CATALOG, args.out)
    else:
        print("Attempting live scrape...", file=sys.stderr)
        live = scrape_live(args.out, args.delay, args.limit)
        if len(live) < 5:
            print(f"Live scrape returned only {len(live)} items. Using embedded catalog instead.", file=sys.stderr)
            save_catalog(EMBEDDED_CATALOG, args.out)
        else:
            save_catalog(live, args.out)