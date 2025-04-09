---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

<table style="width:100%; border:0; border-spacing:0; border-collapse:separate; margin-right:auto; margin-left:auto; font-size:18px;">
  {% for post in site.publications reversed %}
  <tr>
    <td style="border: none; padding:0; width:25%; vertical-align:middle; max-width:100px; max-height:100px;">
  <!-- 이미지 & 토픽 컨테이너 -->
  <div style="display: inline-block; margin:0; padding:0;">
    {% if post.topic %}
      <!-- Topic 배지: 시크한 회색 박스, 바로 위에 붙임 -->
      <div style="background-color:#444; color:#fff; font-style: italic; 
                  border:none; border-radius:0; padding:4px 8px; 
                  margin-bottom:4px; display:inline-block;">
        {{ post.topic }}
      </div>
    {% endif %}
    <!-- 실제 이미지 -->
    <img src="/{{post.image}}" alt="project image" 
         style="display:block; width:auto; height:auto; max-width:100%; margin:0; padding:0;" />
  </div>
</td>
    <td style="border: none; padding:2.5%; width:75%; vertical-align:middle;">
      <h3 style="font-size:22px; margin-bottom:10px;">{{post.title}}</h3>
      <div style="margin-bottom:10px;">{{post.authors}}</div>
      <div style="margin-bottom:10px;"><em>{{post.venue}}</em>, {{ post.date | date: "%Y" }}</div>
      <div style="margin-bottom:10px;">
        {% if post.paper %}
          <a href="{{post.paper}}"
             style="
               font-style: italic;
               background-color: #444;
               color: #fff;
               border: none;
               border-radius: 0;
               padding: 4px 10px;
               text-align: center;
               text-decoration: none;
               display: inline-block;
               margin-top: 8px;
             ">
            paper
          </a>
        {% endif %}
        {% if post.code %}
          <a href="{{post.code}}"
             style="
               font-style: italic;
               background-color: #444;
               color: #fff;
               border: none;
               border-radius: 0;
               padding: 4px 10px;
               text-align: center;
               text-decoration: none;
               display: inline-block;
               margin-left: 4px;
               margin-top: 8px;
             ">
            code
          </a>
        {% endif %}
        {% if post.web %}
          <a href="{{post.web}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">website</a>
        {% endif %}
        {% if post.video %}
          <a href="{{post.video}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">video</a>
        {% endif %}
        {% if post.poster %}
          / <a href="{{post.poster}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">poster</a>
        {% endif %}
        {% if post.slides %}
          / <a href="{{post.slides}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">slides</a>
        {% endif %}
        {% if post.dataset %}
          / <a href="{{post.dataset}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">dataset</a>
        {% endif %}
      </div>
      <div>
        {{ post.excerpt }}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>
