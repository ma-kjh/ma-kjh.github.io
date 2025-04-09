---
permalink: /
title: "Towards Safe and Reliable Artificial Intelligence"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi there👋 My name is Jeonghyeon Kim, and I am a Ph.D. student in Data Science at <a href='https://www.seoultech.ac.kr'>SeoulTech</a>, advised by <a href='https://sites.google.com/ds.seoultech.ac.kr/daintlab/members/director?authuser=0'>Prof. Sangheum Hwang</a> of <a href='https://sites.google.com/ds.seoultech.ac.kr/daintlab/'>DAINTLAB</a>.

 🧗‍♂️I am dedicated to enhancing the reliability and interpretability of AI systems, particularly through:
 
 - Out-of-distribution detection (OoDD)
 - LLM unlearning
 - Energy based models (EBMs)

 💥Ultimate Goal
 
 To develop AI systems that are reliable, and interpretable, ensuring their trustworthy deployment in real-world applications.

## Selected Publications

{% assign selected_publications = site.publications | where: "selected", true %}
{% if selected_publications != empty %}
<ul>
  {% for pub in selected_publications %}
  <li>
    <a href="{{ pub.permalink }}">{{ pub.title }}</a>
    – <em>{{ pub.venue }}</em> ({{ pub.date | date: "%Y" }})
  </li>
  {% endfor %}
</ul>
{% else %}
<p>No selected publications available.</p>
{% endif %}
