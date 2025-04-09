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

<table style="width:100%; border:0; border-spacing:0; border-collapse:separate; margin-right:auto; margin-left:auto; font-size:18px;">
  {% assign selected_publications = site.publications | where: "selected", true | reverse %}
  {% for post in selected_publications %}
  <tr>
    <td style="border: none; padding:2.5%; width:25%; vertical-align:middle; max-width:100px; max-height:100px; position: relative;">
      {% if post.topic %}
        <div style="background-color:#444; color:#fff; font-style: italic; border:none; border-radius:5px; padding:4px 8px; margin-bottom:4px; display:inline-block;">
          {{ post.topic }}
        </div>
      {% endif %}
      <img src="/{{ post.image }}" alt="project image" style="display:block; width:auto; height:auto; max-width:100%;" />
    </td>
    <td style="border: none; padding:2.5%; width:75%; vertical-align:middle;">
      <h3 style="font-size:22px; margin-bottom:10px;">{{ post.title }}</h3>
      <div style="margin-bottom:10px;">{{ post.authors }}</div>
      <div style="margin-bottom:10px;"><em>{{ post.venue }}</em>, {{ post.date | date: "%Y" }}</div>
      <div style="margin-bottom:10px;">
        {% if post.paper %}
          <a href="{{ post.paper }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-top:8px;">
            paper
          </a>
        {% endif %}
        {% if post.code %}
          <a href="{{ post.code }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            code
          </a>
        {% endif %}
        {% if post.web %}
          <a href="{{ post.web }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            website
          </a>
        {% endif %}
        {% if post.video %}
          <a href="{{ post.video }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            video
          </a>
        {% endif %}
        {% if post.poster %}
          / <a href="{{ post.poster }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            poster
          </a>
        {% endif %}
        {% if post.slides %}
          / <a href="{{ post.slides }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            slides
          </a>
        {% endif %}
        {% if post.dataset %}
          / <a href="{{ post.dataset }}" style="font-style: italic; background-color:#444; color:#fff; border:none; border-radius:5px; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px;">
            dataset
          </a>
        {% endif %}
      </div>
      <div>
        {{ post.excerpt }}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>

